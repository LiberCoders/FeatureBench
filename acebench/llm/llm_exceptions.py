"""LLM-related exception definitions."""


class LLMException(Exception):
    """Base exception for LLM-related errors.
    
    When this exception occurs, the pipeline should not persist data_status,
    so the next run can retry.
    """
    pass


class LLMAPIException(LLMException):
    """LLM API call failure."""
    pass


class LLMTimeoutException(LLMException):
    """LLM call timeout."""
    pass


class LLMInitException(LLMException):
    """LLM client initialization failure."""
    pass


class LLMResponseException(LLMException):
    """LLM response parsing failure."""
    pass

