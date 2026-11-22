# A local banking application sandbox

This example runs actual qagents tool calls against a persistent SQLite ledger.
Every account and every unit of value is fictional. `TEST` minor units are
integers; 100 minor units represent one fictional unit. The example makes no
network requests and calls no hosted model or financial service.

It is not a production banking system. Authentication, payment rails, identity
verification, regulatory compliance, fraud detection, encryption, reconciliation
with external institutions, and operational recovery are outside its scope.

## Run the application

From the repository root with qagent-gym and qagents installed:

```powershell
python -m examples.banking_app --db .gym/banking.sqlite3 demo
python -m examples.banking_app --db .gym/banking.sqlite3 accounts
python -m examples.banking_app --db .gym/banking.sqlite3 balance alice-checking
python -m examples.banking_app --db .gym/banking.sqlite3 history alice-checking --limit 10
python -m examples.banking_app --db .gym/banking.sqlite3 transfer alice-checking alice-savings 300 --key savings-intent-1
python -m examples.banking_app --db .gym/banking.sqlite3 --principal bob accounts
```

CLI options `--db` and `--principal` precede the subcommand. The principal selector
simulates an identity supplied by a trusted host; it is not a login or access
control boundary for someone who can run the CLI. In an application, construct
`Principal` from the authenticated session, never from model output or unverified
request parameters. Direct access to the SQLite file is trusted administrative
access in this example.

The default identity is Alice. Initialization creates Alice's checking balance
of 10,000, Alice's savings balance of 2,500, and Bob's checking balance of 5,000
minor units. Initialization is transactional and does not reapply opening funds
when the database is reopened. An unrelated existing SQLite database is refused.

On a fresh database, the demo transfers 1,250 from Alice to Bob, replays the same
intent, and attempts an unauthorized debit from Bob. Alice's checking balance is
then 8,750; Bob's is 6,250. The report includes real tool results, the unauthorized
refusal, account history, and ledger checks. Running `demo` again replays its
stored receipt without another debit. The demo always uses Alice regardless of
`--principal`; other subcommands use the selected identity.

## What executes

`FunctionModel` reads an explicit command and emits a `ToolCallPart`. The normal
qagents Agent loop validates and invokes the tool. The final policy response is
constructed from actual tool results. This is a deterministic command policy,
not a natural-language financial adviser or a measure of LLM performance.

The tools are `list_accounts`, `get_balance`, `get_history`, and `transfer`.
All close over a host-injected `Principal`; none accepts a principal or owner
argument from the model. Read tools expose only owned accounts. Transfers require
ownership of the source and may credit another customer's existing account.
System/opening accounts cannot be used by a customer.

The transfer tool is marked as requiring approval and as having side effects.
The sandbox host explicitly permits it with `Permissions(allow=["transfer"])`.
This permission does not bypass the ledger's ownership, amount, balance or
idempotency checks. A production host would need its own transaction approval
and authenticated identity policy.

## Transaction and replay rules

- Balances are the sum of ledger entries. Every opening funding or customer
  transfer creates two opposite signed entries, together with its transfer and
  receipt record, in one SQLite transaction. The total ledger remains zero.
- Opening funds use a separate system account, whose negative balance offsets
  customer balances. Customer accounts cannot overdraw through this API.
- `BEGIN IMMEDIATE` serializes the idempotency lookup, available-funds check,
  both postings and stored receipt across independent connections. A failure
  during the second posting rolls back the first posting and the receipt record.
- Amounts must be positive integers at most 1,000,000,000 minor units. The tool
  schema and service reject booleans, floats and numeric strings.
- Keys identify a caller's intent and are scoped to the authenticated principal.
  Reusing a committed key with identical source, destination and amount returns
  its original receipt, including the historical source balance. A changed
  payload is refused. Keys remain stored without expiry in this example.
- Rejected validation or insufficient-funds attempts commit nothing and do not
  reserve the key. They may be attempted again; a new deliberate transfer needs
  a new key. Never generate a fresh key merely because a previous response was
  lost.

The application enforces paired postings inside the transaction; SQLite does
not independently enforce a cross-row balancing constraint against arbitrary
direct SQL writes. `audit()` is a trusted local diagnostic, not an agent tool.
Concurrent callers need separate Ledger connections, one per owning thread.
There is no distributed transfer, external payment or crash-recovery claim.

## Tests

```powershell
python -m pytest tests/test_banking_app.py -q
python -m ruff check examples/banking_app.py examples/banking_ledger.py tests/test_banking_app.py
```

Tests exercise real tool execution, unauthorized reads and debits, forged
principal arguments, invalid amounts, conflicting keys, exact receipt replay,
reopening persisted state, and repeated CLI runs. A database trigger injects a
failure between paired postings to check rollback. Parallel connections verify
one posting for a shared key and refusal of one of two competing spends that
would otherwise overdraw the source.

This example is an application fixture for agent/tool testing. It does not train
a banking policy, publish an API server, or connect financial accounts.
