"""Typed case and workflow errors."""


class CaseError(RuntimeError):
    """Base case-layer error."""


class CaseNotFoundError(CaseError):
    """The requested case does not exist."""


class InvalidTransitionError(CaseError):
    """The requested lifecycle transition is not allowed."""


class AttachmentValidationError(CaseError):
    """An uploaded attachment is unsafe or unsupported."""


class ImmutableSnapshotError(CaseError):
    """A submitted assessment snapshot cannot be changed."""
