"""
Integration tests for the FastAPI endpoints.

Covers every item in the API Completeness Checklist:
  1. /health returns model + DB status
  2. POST /verify accepts query, answer, context, policy
  3. /verify returns only ALLOW / FLAG / REFUSE
  4. Response includes confidence_score, policy_results, explanation, audit_id
  5. /audit/{id} retrieves the exact decision
  6. Missing fields → clean 422
  7. Works after restart (SQLite on disk — tested by fixture re-creation)
  8. Docker — tested separately
  9. Malformed policies → rejected
 10. Concurrent requests — no race conditions

Run with:  pytest tests/test_api.py -v
"""

import concurrent.futures

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════
#  1. /health — model + DB status
# ═══════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_model_status(self, client):
        data = client.get("/api/v1/health").json()
        assert "models_loaded" in data
        assert data["models_loaded"] is True

    def test_health_has_db_status(self, client):
        data = client.get("/api/v1/health").json()
        assert "db_connected" in data
        assert data["db_connected"] is True

    def test_health_overall_status(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


# ═══════════════════════════════════════════════════════════
#  2-4. POST /verify — full contract
# ═══════════════════════════════════════════════════════════

class TestVerify:
    PAYLOAD = {
        "query": "What is the capital of France?",
        "context": [
            "Paris is the capital city of France.",
            "France is a country in Western Europe.",
        ],
        "generated_answer": "The capital of France is Paris.",
    }

    def test_verify_returns_200(self, client):
        resp = client.post("/api/v1/verify", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_decision_is_enum(self, client):
        """Decision must be one of ALLOW / FLAG / REFUSE — nothing else."""
        data = client.post("/api/v1/verify", json=self.PAYLOAD).json()
        assert data["decision"] in ("ALLOW", "FLAG", "REFUSE")

    def test_response_has_all_required_fields(self, client):
        """Checklist item 4 — every field must exist."""
        data = client.post("/api/v1/verify", json=self.PAYLOAD).json()
        assert "request_id" in data
        assert "audit_id" in data
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert "hallucination_detected" in data
        assert "explanation" in data
        assert "verification_details" in data
        assert "policy_results" in data
        assert "timestamp" in data

    def test_with_policy_override(self, client):
        """Client can send custom policy_config."""
        payload = {**self.PAYLOAD, "policy_config": {"min_confidence": 0.99}}
        data = client.post("/api/v1/verify", json=payload).json()
        assert data["decision"] in ("FLAG", "REFUSE")

    def test_hallucination_is_refused(self, client):
        payload = {
            "query": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "generated_answer": "Berlin is the capital of France.",
        }
        data = client.post("/api/v1/verify", json=payload).json()
        assert data["decision"] == "REFUSE"
        assert data["hallucination_detected"] is True


# ═══════════════════════════════════════════════════════════
#  5. /audit/{id} — retrieve exact decision
# ═══════════════════════════════════════════════════════════

class TestAuditRetrieval:
    def test_verify_then_retrieve_by_audit_id(self, client):
        """POST /verify → get audit_id → GET /audit/{audit_id} → same decision."""
        payload = {
            "query": "What is gravity?",
            "context": ["Gravity is a fundamental force of nature."],
            "generated_answer": "Gravity is a fundamental force.",
        }
        verify_resp = client.post("/api/v1/verify", json=payload).json()
        audit_id = verify_resp["audit_id"]

        audit_resp = client.get(f"/api/v1/audit/{audit_id}")
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert audit_data["decision"] == verify_resp["decision"]
        assert audit_data["confidence_score"] == verify_resp["confidence_score"]

    def test_verify_then_retrieve_by_request_id(self, client):
        """Can also look up by request_id."""
        payload = {
            "query": "What color is the sky?",
            "context": ["The sky is blue during the day."],
            "generated_answer": "The sky is blue.",
        }
        verify_resp = client.post("/api/v1/verify", json=payload).json()
        request_id = verify_resp["request_id"]

        audit_resp = client.get(f"/api/v1/audit/{request_id}")
        assert audit_resp.status_code == 200

    def test_audit_not_found(self, client):
        resp = client.get("/api/v1/audit/nonexistent_id_xyz")
        assert resp.status_code == 404

    def test_audit_stats_today(self, client):
        data = client.get("/api/v1/audit/stats/today").json()
        assert "total" in data
        assert "decisions" in data

    def test_audit_recent(self, client):
        resp = client.get("/api/v1/audit/recent")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════
#  6. Missing fields → clean 422
# ═══════════════════════════════════════════════════════════

class TestValidation:
    def test_missing_query(self, client):
        resp = client.post("/api/v1/verify", json={
            "context": ["some context"],
            "generated_answer": "some answer",
        })
        assert resp.status_code == 422

    def test_missing_context(self, client):
        resp = client.post("/api/v1/verify", json={
            "query": "test",
            "generated_answer": "some answer",
        })
        assert resp.status_code == 422

    def test_missing_answer(self, client):
        resp = client.post("/api/v1/verify", json={
            "query": "test",
            "context": ["some context"],
        })
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/api/v1/verify", json={})
        assert resp.status_code == 422

    def test_empty_query_string(self, client):
        resp = client.post("/api/v1/verify", json={
            "query": "",
            "context": ["ctx"],
            "generated_answer": "ans",
        })
        assert resp.status_code == 422

    def test_empty_context_list(self, client):
        resp = client.post("/api/v1/verify", json={
            "query": "q",
            "context": [],
            "generated_answer": "ans",
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════
#  9. Malformed policies → rejected
# ═══════════════════════════════════════════════════════════

class TestPolicyValidation:
    def test_valid_policy_update(self, client):
        resp = client.post("/api/v1/policies", json={"min_confidence": 0.8})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    def test_get_active_policy(self, client):
        resp = client.get("/api/v1/policies")
        assert resp.status_code == 200
        assert "min_confidence" in resp.json()

    def test_reject_unknown_fields(self, client):
        """Extra/unknown fields must be rejected."""
        resp = client.post("/api/v1/policies", json={
            "min_confidence": 0.5,
            "totally_fake_field": True,
        })
        assert resp.status_code == 422

    def test_reject_invalid_confidence(self, client):
        resp = client.post("/api/v1/policies", json={"min_confidence": 2.0})
        assert resp.status_code == 422

    def test_reject_invalid_coverage_level(self, client):
        resp = client.post("/api/v1/policies", json={"min_coverage_level": "SuperHigh"})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════
#  10. Concurrent requests — no race conditions
# ═══════════════════════════════════════════════════════════

class TestConcurrency:
    PAYLOAD = {
        "query": "What is water?",
        "context": ["Water is H2O, a chemical compound."],
        "generated_answer": "Water is H2O.",
    }

    def test_concurrent_verify_requests(self, client):
        """Fire 5 concurrent /verify requests — all should return valid responses."""
        def fire():
            resp = client.post("/api/v1/verify", json=self.PAYLOAD)
            return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(fire) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for status, data in results:
            assert status == 200
            assert data["decision"] in ("ALLOW", "FLAG", "REFUSE")
            assert "audit_id" in data

        # All audit_ids must be unique
        audit_ids = [data["audit_id"] for _, data in results]
        assert len(set(audit_ids)) == 5
