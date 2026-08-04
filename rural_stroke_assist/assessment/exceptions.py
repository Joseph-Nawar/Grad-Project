"""Typed failures raised by the assessment service."""


class AssessmentError(RuntimeError):
    """Base class for application-level assessment failures."""


class AssessmentInputError(AssessmentError):
    """The supplied assessment input cannot be validated."""


class AssessmentExecutionError(AssessmentError):
    """An unexpected programming or runtime error occurred during execution."""
