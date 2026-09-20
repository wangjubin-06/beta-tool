"""Data-access errors with messages a user can act on.

Put this in data_collection/ and use it from TiingoApi, FredApi and FrenchApi.

Typical use after a request:

    check_response(resp, provider=..., var=..., what=ticker)   # HTTP errors (4xx/5xx)
    check_body(resp, provider=..., var=..., what=ticker)       # errors hidden in a 200 (CSV endpoints)
    # or, for JSON endpoints:
    check_payload(resp.json(), provider=..., var=..., what=ticker)
"""


class DataError(Exception):
    """Base class for data-fetching problems the user can fix."""


class MissingApiKeyError(DataError):
    pass


class InvalidApiKeyError(DataError):
    pass


class TickerNotFoundError(DataError):
    pass


class RateLimitError(DataError):
    pass


_SETUP_HINT = (
    "Set it in your shell, for example:\n"
    '  PowerShell:   $env:{var} = "your-key"\n'
    '  macOS/Linux:  export {var}="your-key"'
)


def require_api_key(key: str | None, var: str, provider: str) -> str:
    """Return a cleaned key, or raise a clear error if it is missing or blank.

    Call this in the client's __init__, before any network request.
    """
    key = (key or "").strip().strip("\"'")
    if not key:
        raise MissingApiKeyError(
            f"{provider} API key not found: environment variable {var} is not set.\n"
            + _SETUP_HINT.format(var=var)
        )
    return key


def _raise_from_message(message: str, *, provider: str, var: str, what: str) -> None:
    """Map a provider's error message onto the most specific DataError.

    The keyword matching is a best guess. Anything unrecognised still raises a
    generic DataError that includes the provider's own message.
    """
    low = message.lower()

    if any(w in low for w in ("token", "api key", "api_key", "not registered", "authenticat", "credential")):
        raise InvalidApiKeyError(
            f"{provider} rejected your API key: {message}\n"
            f"Check that {var} is correct and active."
        )
    if "not found" in low or "does not exist" in low:
        raise TickerNotFoundError(
            f"{provider} has no data for {what or 'this request'}: {message}"
        )
    if any(w in low for w in ("limit", "throttl", "exceed", "allocation", "run over")):
        raise RateLimitError(f"{provider} rate limit reached: {message}")

    raise DataError(f"{provider} returned an error: {message}")


def check_payload(data, *, provider: str, var: str, what: str = "") -> None:
    """Catch error objects that arrive as parsed JSON (e.g. with HTTP 200).

    Successful price endpoints return a list of records. An error comes back as
    a JSON object carrying a message such as {"detail": "..."}. Call this right
    after resp.json(), before building a DataFrame from it.
    """
    if not isinstance(data, dict):
        return

    detail = data.get("detail") or data.get("message") or data.get("error") or data.get("error_message")
    if not detail:
        return  # a plain dict without an error message, e.g. a metadata endpoint

    _raise_from_message(str(detail), provider=provider, var=var, what=what)


def check_body(resp, *, provider: str, var: str, what: str = "") -> None:
    """Catch errors hidden in the body of a response that should be CSV.

    A valid CSV starts with a header row, so it never begins with '{' or
    'Error:'. A JSON error object begins with '{', and some APIs send plain
    text starting with 'Error:'. Sniffing the text means resp.json() is only
    called when the body looks like JSON, so a good CSV never triggers a JSON
    decode error.
    """
    body = resp.text.strip()

    if body.lower().startswith("error:"):
        _raise_from_message(body, provider=provider, var=var, what=what)

    if body.startswith("{"):
        try:
            data = resp.json()
        except ValueError:
            return
        check_payload(data, provider=provider, var=var, what=what)


def check_response(resp, *, provider: str, var: str, what: str = "") -> None:
    """Translate HTTP-level failures into DataError subclasses.

    Call this immediately after each request.
    """
    code = resp.status_code

    if code in (401, 403):
        raise InvalidApiKeyError(
            f"{provider} rejected the request (HTTP {code}). Your API key in {var} may be "
            "invalid or inactive, or your plan may not allow this data."
        )
    if code == 404:
        raise TickerNotFoundError(
            f"{provider} has no data for {what or 'this request'} (HTTP 404). "
            "Check the ticker symbol."
        )
    if code == 429:
        raise RateLimitError(
            f"{provider} rate limit reached (HTTP 429). Wait a while, or request fewer tickers."
        )

    if code >= 400:
        # Some APIs (e.g. FRED) report bad keys and unknown series as HTTP 400
        # with a message in the body, so read that before the generic fallback.
        check_body(resp, provider=provider, var=var, what=what)

        # Deliberately not resp.raise_for_status(): its message includes the full
        # URL, which may contain the API key.
        raise DataError(f"{provider} returned HTTP {code}: {resp.text[:200].strip()}")