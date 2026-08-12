"""Minimal Cognito pre-token trigger for facility-scope propagation."""


def lambda_handler(event, _context):
    attributes = event.get("request", {}).get("userAttributes", {})
    raw_scope = attributes.get("custom:facility_scope", "")
    facilities = [item.strip() for item in raw_scope.split(",") if item.strip()]
    if facilities:
        response = event.setdefault("response", {})
        overrides = response.setdefault("claimsAndScopeOverrideDetails", {})
        access = overrides.setdefault("accessTokenGeneration", {})
        claims = access.setdefault("claimsToAddOrOverride", {})
        claims["facilities"] = facilities
    return event
