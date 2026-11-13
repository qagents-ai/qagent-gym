"""A fictional local ledger. Authentication and payment rails are deliberately absent."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

MAX_AMOUNT = 1_000_000_000
CURRENCY = "TEST"
SCHEMA = "qagent-gym-banking-v1"


class BankError(ValueError):
    """A rejected sandbox operation; no transfer was committed."""


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise BankError(f"{name} must be a nonempty string of at most 128 characters")


@dataclass(frozen=True)
class Principal:
    """Identity supplied by trusted application code, never by a model tool argument."""

    subject: str

    def __post_init__(self) -> None:
        identifier(self.subject, "principal")
        if self.subject.startswith("__"):
            raise BankError("reserved principal")


class Ledger:
    """One SQLite connection per owning thread, with serializable transfer writes.

    Every posting has two opposite integer entries in one database transaction.
    Balances are derived from entries. BEGIN IMMEDIATE serializes the balance check,
    idempotency check and both postings across independent connections.
    """

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path), timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        try:
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                tables = {
                    row[0]
                    for row in self._db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if tables:
                    if "bank_metadata" not in tables:
                        raise BankError("this is not a banking sandbox database")
                    row = self._db.execute(
                        "SELECT value FROM bank_metadata WHERE key='schema'"
                    ).fetchone()
                    if row is None or row[0] != SCHEMA:
                        raise BankError("unsupported banking sandbox database schema")
                else:
                    self._create_schema()
        except BaseException:
            self._db.close()
            raise

    def _create_schema(self) -> None:
        # Execute separately: executescript would implicitly commit the transaction.
        statements = [
            "CREATE TABLE bank_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            """CREATE TABLE bank_accounts (
                account_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, label TEXT NOT NULL,
                is_system INTEGER NOT NULL CHECK (is_system IN (0, 1)))""",
            f"""CREATE TABLE bank_transfers (
                transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                principal_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                source_id TEXT NOT NULL REFERENCES bank_accounts(account_id),
                destination_id TEXT NOT NULL REFERENCES bank_accounts(account_id),
                amount_minor INTEGER NOT NULL
                    CHECK (amount_minor > 0 AND amount_minor <= {MAX_AMOUNT}),
                kind TEXT NOT NULL CHECK (kind IN ('opening', 'transfer')),
                receipt TEXT NOT NULL,
                UNIQUE (principal_id, idempotency_key), CHECK (source_id != destination_id))""",
            """CREATE TABLE bank_entries (
                transfer_id INTEGER NOT NULL REFERENCES bank_transfers(transfer_id),
                account_id TEXT NOT NULL REFERENCES bank_accounts(account_id),
                amount_minor INTEGER NOT NULL CHECK (amount_minor != 0),
                PRIMARY KEY (transfer_id, account_id))""",
            "CREATE INDEX bank_entries_account ON bank_entries(account_id, transfer_id)",
        ]
        for statement in statements:
            self._db.execute(statement)
        self._db.execute("INSERT INTO bank_metadata VALUES ('schema', ?)", (SCHEMA,))

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *args: object) -> None:
        self._db.close()

    def seed_demo(self) -> None:
        """Trusted fixture setup, deliberately unavailable as an agent tool."""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self._db.execute("SELECT 1 FROM bank_metadata WHERE key='seeded'").fetchone():
                return
            self._db.execute(
                "INSERT INTO bank_accounts VALUES (?, ?, ?, 1)",
                ("__opening__", "__system__", "Fictional opening funds"),
            )
            for account_id, owner, label, amount in (
                ("alice-checking", "alice", "Alice checking", 10_000),
                ("alice-savings", "alice", "Alice savings", 2_500),
                ("bob-checking", "bob", "Bob checking", 5_000),
            ):
                self._db.execute(
                    "INSERT INTO bank_accounts VALUES (?, ?, ?, 0)", (account_id, owner, label)
                )
                self._post("__system__", account_id, "__opening__", account_id, amount, "opening")
            self._db.execute("INSERT INTO bank_metadata VALUES ('seeded', 'yes')")

    def _balance(self, account_id: str) -> int:
        return self._db.execute(
            "SELECT COALESCE(SUM(amount_minor), 0) FROM bank_entries WHERE account_id=?",
            (account_id,),
        ).fetchone()[0]

    def _owned(self, principal: Principal, account_id: str) -> sqlite3.Row:
        if not isinstance(principal, Principal):
            raise BankError("an authenticated Principal is required")
        identifier(account_id, "account_id")
        row = self._db.execute(
            "SELECT * FROM bank_accounts WHERE account_id=?", (account_id,)
        ).fetchone()
        if row is None or row["is_system"] or row["owner_id"] != principal.subject:
            raise BankError("account is not accessible to this principal")
        return row

    def accounts(self, principal: Principal) -> list[dict]:
        if not isinstance(principal, Principal):
            raise BankError("an authenticated Principal is required")
        # A single SELECT gives a consistent ownership/balance snapshot.
        return [
            {**dict(row), "currency": CURRENCY}
            for row in self._db.execute(
                """SELECT a.account_id, a.label, COALESCE(SUM(e.amount_minor), 0) AS balance_minor
                FROM bank_accounts a LEFT JOIN bank_entries e USING (account_id)
                WHERE a.owner_id=? AND a.is_system=0 GROUP BY a.account_id ORDER BY a.account_id""",
                (principal.subject,),
            )
        ]

    def balance(self, principal: Principal, account_id: str) -> dict:
        with self._db:
            self._db.execute("BEGIN")
            account = self._owned(principal, account_id)
            return {
                "account_id": account_id,
                "label": account["label"],
                "balance_minor": self._balance(account_id),
                "currency": CURRENCY,
            }

    def history(self, principal: Principal, account_id: str, *, limit: int = 20) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise BankError("limit must be an integer between 1 and 100")
        with self._db:
            self._db.execute("BEGIN")
            self._owned(principal, account_id)
            return [
                {**dict(row), "currency": CURRENCY}
                for row in self._db.execute(
                    """SELECT t.transfer_id, t.kind, e.amount_minor,
                        CASE WHEN t.source_id=? THEN t.destination_id ELSE t.source_id END
                            AS counterparty_account
                        FROM bank_entries e JOIN bank_transfers t USING (transfer_id)
                        WHERE e.account_id=? ORDER BY t.transfer_id DESC LIMIT ?""",
                    (account_id, account_id, limit),
                )
            ]

    def transfer(
        self,
        principal: Principal,
        *,
        source_id: str,
        destination_id: str,
        amount_minor: int,
        idempotency_key: str,
    ) -> dict:
        """Move fictional funds; replay returns the exact original receipt.

        The key belongs to the authenticated principal and identifies one intent.
        A failed validation/funds check does not reserve a key. A successful key
        remains in this sandbox database without expiry.
        """
        if type(amount_minor) is not int or not 1 <= amount_minor <= MAX_AMOUNT:
            raise BankError(f"amount_minor must be an integer between 1 and {MAX_AMOUNT}")
        identifier(idempotency_key, "idempotency_key")
        identifier(destination_id, "destination_id")
        if source_id == destination_id:
            raise BankError("source and destination must differ")
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            self._owned(principal, source_id)
            recipient = self._db.execute(
                "SELECT is_system FROM bank_accounts WHERE account_id=?", (destination_id,)
            ).fetchone()
            if recipient is None or recipient[0]:
                raise BankError("destination account is unavailable")
            prior = self._db.execute(
                """SELECT * FROM bank_transfers WHERE principal_id=? AND idempotency_key=?""",
                (principal.subject, idempotency_key),
            ).fetchone()
            if prior:
                if (prior["source_id"], prior["destination_id"], prior["amount_minor"]) != (
                    source_id,
                    destination_id,
                    amount_minor,
                ):
                    raise BankError("idempotency_key was already used for a different transfer")
                return json.loads(prior["receipt"])
            if self._balance(source_id) < amount_minor:
                raise BankError("insufficient fictional funds")
            return self._post(
                principal.subject,
                idempotency_key,
                source_id,
                destination_id,
                amount_minor,
                "transfer",
            )

    def _post(
        self, principal: str, key: str, source: str, destination: str, amount: int, kind: str
    ) -> dict:
        cursor = self._db.execute(
            """INSERT INTO bank_transfers
                (principal_id,idempotency_key,source_id,destination_id,amount_minor,kind,receipt)
                VALUES (?, ?, ?, ?, ?, ?, '{}')""",
            (principal, key, source, destination, amount, kind),
        )
        transfer_id = cursor.lastrowid
        self._db.executemany(
            "INSERT INTO bank_entries VALUES (?, ?, ?)",
            [(transfer_id, source, -amount), (transfer_id, destination, amount)],
        )
        receipt = {
            "transfer_id": transfer_id,
            "source_id": source,
            "destination_id": destination,
            "amount_minor": amount,
            "currency": CURRENCY,
            "source_balance_minor_after": self._balance(source),
        }
        self._db.execute(
            "UPDATE bank_transfers SET receipt=? WHERE transfer_id=?",
            (json.dumps(receipt, sort_keys=True), transfer_id),
        )
        return receipt

    def audit(self) -> dict:
        """Trusted local diagnostic, not an agent tool or customer-facing endpoint."""
        with self._db:
            self._db.execute("BEGIN")
            malformed = self._db.execute(
                """SELECT t.transfer_id FROM bank_transfers t
                LEFT JOIN bank_entries e USING (transfer_id) GROUP BY t.transfer_id
                HAVING COUNT(e.account_id) != 2 OR SUM(e.amount_minor) != 0"""
            ).fetchall()
            overdrafts = self._db.execute(
                """SELECT a.account_id FROM bank_accounts a
                LEFT JOIN bank_entries e USING (account_id) WHERE a.is_system=0
                GROUP BY a.account_id HAVING COALESCE(SUM(e.amount_minor),0) < 0"""
            ).fetchall()
            return {
                "balanced": not malformed,
                "customer_overdrafts": len(overdrafts),
                "ledger_total_minor": self._db.execute(
                    "SELECT COALESCE(SUM(amount_minor),0) FROM bank_entries"
                ).fetchone()[0],
                "customer_transfers": self._db.execute(
                    "SELECT COUNT(*) FROM bank_transfers WHERE kind='transfer'"
                ).fetchone()[0],
            }
