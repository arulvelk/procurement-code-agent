PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS classifications (
    classification_id TEXT PRIMARY KEY,
    taxonomy          TEXT NOT NULL,
    description       TEXT NOT NULL,
    code              TEXT NOT NULL,
    title             TEXT NOT NULL,
    confidence        REAL NOT NULL,
    rationale         TEXT NOT NULL,
    evidence_json     TEXT NOT NULL,
    alternatives_json TEXT NOT NULL,
    decided_by        TEXT NOT NULL,
    latency_ms        INTEGER NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id         TEXT PRIMARY KEY,
    classification_id TEXT NOT NULL REFERENCES classifications(classification_id),
    decision          TEXT NOT NULL CHECK (decision IN ('accept', 'reject', 'reassign')),
    reviewer          TEXT NOT NULL,
    assigned_code     TEXT,
    note              TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reviews_classification
    ON reviews (classification_id);

-- The key is scoped by subject and operation, not global. Two callers that both
-- happen to use the key "retry-1" are making two different requests, and a
-- global key space would silently hand the second one the first one's answer.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    subject             TEXT NOT NULL,
    operation           TEXT NOT NULL,
    key                 TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    state               TEXT NOT NULL CHECK (state IN ('in_flight', 'completed')),
    response_json       TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (subject, operation, key)
);
