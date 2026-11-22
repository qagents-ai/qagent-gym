import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from examples.banking_app import main, run_action, run_demo
from examples.banking_ledger import MAX_AMOUNT, BankError, Ledger, Principal

ALICE = Principal("alice")
BOB = Principal("bob")


def transfer(ledger, key="intent-1", amount=1250, **overrides):
    arguments = {
        "source_id": "alice-checking",
        "destination_id": "bob-checking",
        "amount_minor": amount,
        "idempotency_key": key,
    }
    return ledger.transfer(ALICE, **{**arguments, **overrides})


def balances(ledger):
    return {
        account["account_id"]: account["balance_minor"]
        for principal in (ALICE, BOB)
        for account in ledger.accounts(principal)
    }


def test_balanced_persistent_ledger_and_idempotent_receipt(tmp_path):
    path = tmp_path / "bank.sqlite3"
    with Ledger(path) as ledger:
        ledger.seed_demo()
        ledger.seed_demo()
        assert balances(ledger) == {
            "alice-checking": 10000,
            "alice-savings": 2500,
            "bob-checking": 5000,
        }
        receipt = transfer(ledger)
        assert receipt["source_balance_minor_after"] == 8750
        assert transfer(ledger) == receipt
        transfer(ledger, "different-intent", 100, destination_id="alice-savings")
        assert transfer(ledger) == receipt  # historical receipt, not the current balance
        assert ledger.balance(ALICE, "alice-checking")["balance_minor"] == 8650
        assert ledger.history(ALICE, "alice-checking")[0]["amount_minor"] == -100
        assert ledger.audit() == {
            "balanced": True,
            "customer_overdrafts": 0,
            "ledger_total_minor": 0,
            "customer_transfers": 2,
        }
    with Ledger(path) as ledger:
        ledger.seed_demo()  # opening balances are never re-applied
        assert transfer(ledger) == receipt
        assert balances(ledger) == {
            "alice-checking": 8650,
            "alice-savings": 2600,
            "bob-checking": 6250,
        }


@pytest.mark.parametrize("amount", [0, -1, 1.5, True, "100", MAX_AMOUNT + 1])
def test_invalid_amount_cannot_change_balances(tmp_path, amount):
    with Ledger(tmp_path / "bank.sqlite3") as ledger:
        ledger.seed_demo()
        before = balances(ledger)
        with pytest.raises(BankError, match="amount_minor"):
            transfer(ledger, amount=amount)
        assert balances(ledger) == before
        assert ledger.audit()["customer_transfers"] == 0


def test_owner_checks_idempotency_conflicts_and_destination_checks(tmp_path):
    with Ledger(tmp_path / "bank.sqlite3") as ledger:
        ledger.seed_demo()
        receipt = transfer(ledger)
        before = balances(ledger)
        for operation in (ledger.balance, ledger.history):
            with pytest.raises(BankError, match="not accessible"):
                operation(ALICE, "bob-checking")
        with pytest.raises(BankError, match="not accessible"):
            ledger.transfer(
                BOB,
                source_id="alice-checking",
                destination_id="bob-checking",
                amount_minor=1250,
                idempotency_key="intent-1",
            )
        for overrides in (
            {"amount_minor": 100},
            {"destination_id": "alice-savings"},
            {"source_id": "alice-savings"},
        ):
            arguments = {
                "source_id": "alice-checking",
                "destination_id": "bob-checking",
                "amount_minor": 1250,
                "idempotency_key": "intent-1",
            }
            with pytest.raises(BankError, match="different transfer"):
                ledger.transfer(ALICE, **{**arguments, **overrides})
        for destination in ("missing", "__opening__"):
            with pytest.raises(BankError, match="destination"):
                transfer(ledger, "other", destination_id=destination)
        with pytest.raises(BankError, match="must differ"):
            transfer(ledger, "other", destination_id="alice-checking")
        with pytest.raises(BankError, match="insufficient"):
            transfer(ledger, "poor", amount=10000)
        assert transfer(ledger) == receipt
        assert balances(ledger) == before
        assert ledger.audit()["customer_transfers"] == 1


def test_second_posting_failure_rolls_back_receipt_and_both_entries(tmp_path):
    path = tmp_path / "bank.sqlite3"
    with Ledger(path) as ledger:
        ledger.seed_demo()
        before = balances(ledger)
        with sqlite3.connect(path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_credit BEFORE INSERT ON bank_entries
                WHEN NEW.account_id='bob-checking' AND NEW.amount_minor > 0
                BEGIN SELECT RAISE(ABORT, 'injected posting failure'); END
            """)
        with pytest.raises(sqlite3.IntegrityError, match="injected posting failure"):
            transfer(ledger)
        assert balances(ledger) == before
        assert ledger.audit()["customer_transfers"] == 0
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER reject_credit")
        receipt = transfer(
            ledger
        )  # same key can succeed because the failed transaction rolled back
        assert receipt["amount_minor"] == 1250
        assert ledger.audit()["balanced"] is True


def test_concurrent_same_key_posts_one_pair(tmp_path):
    path = tmp_path / "bank.sqlite3"
    with Ledger(path) as ledger:
        ledger.seed_demo()
    ready = threading.Barrier(4)

    def post(_):
        with Ledger(path) as ledger:
            ready.wait(timeout=10)
            return transfer(ledger, "concurrent-intent", 6000)

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(post, range(4)))
    assert all(receipt == receipts[0] for receipt in receipts)
    with Ledger(path) as ledger:
        assert ledger.balance(ALICE, "alice-checking")["balance_minor"] == 4000
        assert ledger.balance(BOB, "bob-checking")["balance_minor"] == 11000
        assert ledger.audit()["customer_transfers"] == 1
        assert ledger.audit()["balanced"] is True


def test_concurrent_spends_cannot_overdraw(tmp_path):
    path = tmp_path / "bank.sqlite3"
    with Ledger(path) as ledger:
        ledger.seed_demo()
    ready = threading.Barrier(2)

    def post(index):
        with Ledger(path) as ledger:
            ready.wait(timeout=10)
            try:
                transfer(ledger, f"spend-{index}", 8000)
            except BankError as exc:
                return str(exc)
            return "posted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(post, range(2)))
    assert sorted(outcomes) == ["insufficient fictional funds", "posted"]
    with Ledger(path) as ledger:
        assert ledger.balance(ALICE, "alice-checking")["balance_minor"] == 2000
        assert ledger.audit() == {
            "balanced": True,
            "customer_overdrafts": 0,
            "ledger_total_minor": 0,
            "customer_transfers": 1,
        }


def test_actual_qagents_tools_execute_and_cannot_impersonate(tmp_path):
    with Ledger(tmp_path / "bank.sqlite3") as ledger:
        ledger.seed_demo()
        report = asyncio.run(run_demo(ledger))
        assert report["transfer"][0]["name"] == "transfer"
        assert report["transfer"][0]["is_error"] is False
        assert report["transfer"][0]["content"]["amount_minor"] == 1250
        assert report["replay_matches"] is True
        assert report["unauthorized_attempt"][0]["is_error"] is True
        assert report["audit"]["customer_transfers"] == 1
        before = balances(ledger)
        for amount in (0, -1, True, 1.5, "100"):
            result = asyncio.run(
                run_action(
                    ledger,
                    ALICE,
                    "transfer",
                    {
                        "source_id": "alice-checking",
                        "destination_id": "bob-checking",
                        "amount_minor": amount,
                        "idempotency_key": "invalid-amount",
                    },
                )
            )
            assert result["tool_calls"][0]["name"] == "transfer"
            assert result["tool_results"][0]["is_error"] is True
            assert balances(ledger) == before
        result = asyncio.run(
            run_action(
                ledger,
                ALICE,
                "transfer",
                {
                    "source_id": "bob-checking",
                    "destination_id": "alice-checking",
                    "amount_minor": 100,
                    "idempotency_key": "forged-principal",
                    "principal": "bob",
                },
            )
        )
        assert result["tool_results"][0]["is_error"] is True
        assert balances(ledger) == before
        accounts = asyncio.run(run_action(ledger, BOB, "list_accounts"))
        assert [row["account_id"] for row in accounts["tool_results"][0]["content"]] == [
            "bob-checking"
        ]
        denied = asyncio.run(
            run_action(ledger, BOB, "get_balance", {"account_id": "alice-checking"})
        )
        assert denied["tool_results"][0]["is_error"] is True
        again = asyncio.run(run_demo(ledger))
        assert again["audit"]["customer_transfers"] == 1


def test_database_identity_and_cli(tmp_path, capsys):
    unrelated = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(BankError, match="not a banking"):
        Ledger(unrelated)
    with sqlite3.connect(unrelated) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == [("unrelated",)]
    path = str(tmp_path / "bank.sqlite3")
    assert main(["--db", path, "demo"]) == 0
    assert json.loads(capsys.readouterr().out)["audit"]["customer_transfers"] == 1
    assert main(["--db", path, "demo"]) == 0
    assert json.loads(capsys.readouterr().out)["audit"]["customer_transfers"] == 1
    assert main(["--db", path, "--principal", "bob", "balance", "alice-checking"]) == 1
    assert json.loads(capsys.readouterr().out)["tool_results"][0]["is_error"] is True
