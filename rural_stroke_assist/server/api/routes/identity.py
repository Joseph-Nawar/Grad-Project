from fastapi import APIRouter, Depends

from rural_stroke_assist.server.api.dependencies import get_principal
from rural_stroke_assist.server.api.schemas.common import IdentityResponse
from rural_stroke_assist.server.principal import Principal

router = APIRouter(prefix="/api/v1/identity", tags=["identity"])


@router.get("/me", operation_id="identity_me", response_model=IdentityResponse)
def identity_me(principal: Principal = Depends(get_principal)) -> IdentityResponse:
    return IdentityResponse(subject=principal.subject, roles=sorted(principal.roles), facilities=sorted(principal.facilities))


@router.get("/roles", operation_id="identity_roles", response_model=IdentityResponse)
def identity_roles(principal: Principal = Depends(get_principal)) -> IdentityResponse:
    return identity_me(principal)
