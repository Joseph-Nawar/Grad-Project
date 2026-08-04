"""Typed adapter failures."""


class AdapterError(RuntimeError):
    """Base class for expected adapter failures."""


class ArtifactConfigurationError(AdapterError):
    """The canonical artifact cannot be configured or loaded."""


class FeatureContractError(AdapterError):
    """Input or extracted features do not match the trained contract."""


class InferenceFailure(AdapterError):
    """The configured model failed while performing inference."""
