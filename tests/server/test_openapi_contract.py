from __future__ import annotations

from rural_stroke_assist.server.app import create_app


def test_openapi_business_paths_are_versioned_and_operations_unique() -> None:
    document = create_app(initialize_resources=False).openapi()
    assert "/health/live" in document["paths"]
    assert "/api/v1/cases" in document["paths"]
    assert "/api/v1/health/live" not in document["paths"]
    operation_ids = [
        operation["operationId"]
        for path in document["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))
    assert all(
        path.startswith("/api/v1/") or path in {"/health/live", "/health/ready", "/version"}
        for path in document["paths"]
    )


def test_secured_business_operations_declare_bearer_security() -> None:
    document = create_app(initialize_resources=False).openapi()
    secured = []
    for path, operations in document["paths"].items():
        if path.startswith("/api/v1/"):
            secured.extend(
                operation for operation in operations.values() if isinstance(operation, dict)
            )
    assert secured
    assert all(operation.get("security") for operation in secured)


def test_openapi_exposes_stable_errors_and_required_retry_headers() -> None:
    document = create_app(initialize_resources=False).openapi()
    assert "HTTPValidationError" not in document["components"]["schemas"]
    assert "ValidationError" not in document["components"]["schemas"]
    error_schema = document["components"]["schemas"]["ErrorResponse"]
    assert set(error_schema["properties"]) == {"code", "message", "details", "correlation_id"}
    assert set(error_schema["required"]) == {"code", "message", "details", "correlation_id"}
    required_headers = {
        "case_create": {"Idempotency-Key"},
        "assessment_create": {"Idempotency-Key"},
        "case_submit": {"Idempotency-Key", "If-Match"},
        "review_claim": {"Idempotency-Key"},
        "review_create": {"Idempotency-Key"},
        "case_update": {"If-Match"},
    }
    for path_item in document["paths"].values():
        for operation in path_item.values():
            if (
                not isinstance(operation, dict)
                or operation.get("operationId") not in required_headers
            ):
                continue
            headers = {
                parameter["name"]
                for parameter in operation.get("parameters", [])
                if parameter.get("in") == "header" and parameter.get("required")
            }
            assert required_headers[operation["operationId"]].issubset(headers)
            if "If-Match" in required_headers[operation["operationId"]]:
                if_match = next(
                    parameter
                    for parameter in operation["parameters"]
                    if parameter["name"] == "If-Match"
                )
                assert if_match["schema"] == {"type": "string"}
    assert "503" in document["paths"]["/health/ready"]["get"]["responses"]
    attachment_response = document["paths"]["/api/v1/attachments/{attachment_id}"]["get"][
        "responses"
    ]["200"]
    assert any(
        content.get("schema", {}).get("format") == "binary"
        for content in attachment_response["content"].values()
    )
    import_operation = document["paths"]["/api/v1/assessments/import"]["post"]
    assert import_operation["summary"] == "Import a collector-generated assessment"
    assert any(
        parameter["name"] == "Idempotency-Key" and parameter["required"]
        for parameter in import_operation["parameters"]
    )
    attachment_upload = document["paths"]["/api/v1/attachments"]["post"]
    assert attachment_upload["summary"] == "Upload an attachment"
    assert any(
        parameter["name"] == "Idempotency-Key" and not parameter["required"]
        for parameter in attachment_upload["parameters"]
    )


def test_openapi_uses_typed_ml_input_schemas_and_route_error_contracts() -> None:
    document = create_app(initialize_resources=False).openapi()
    assessment_schema = document["components"]["schemas"]["AssessmentCreateRequest"]
    assert assessment_schema["properties"]["id"]["type"] == "string"
    metadata_schema = assessment_schema["properties"]["metadata"]
    symptoms_schema = assessment_schema["properties"]["acute_symptoms"]
    assert any(
        item.get("$ref", "").endswith("MetadataInputSchema") for item in metadata_schema["anyOf"]
    )
    assert any(
        item.get("$ref", "").endswith("AcuteSymptomsSchema") for item in symptoms_schema["anyOf"]
    )
    assert document["paths"]["/api/v1/cases/{case_id}"]["patch"]["responses"]["428"]
    assert document["paths"]["/api/v1/attachments"]["post"]["responses"]["413"]
    assert document["paths"]["/api/v1/attachments"]["post"]["responses"]["415"]
