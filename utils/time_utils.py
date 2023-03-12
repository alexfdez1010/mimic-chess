from utils.constants import SECONDS_IN_HOUR, SECONDS_IN_MINUTE


def time_string_to_seconds(time: str) -> int:
    """
    Converts a time in string format to seconds
    :param time: time in string format
    :return: time in seconds
    """
    hours, minutes, seconds = list(map(int, time.split(":")))
    return hours * SECONDS_IN_HOUR + minutes * SECONDS_IN_MINUTE + seconds


def seconds_to_time(seconds: int) -> str:
    """
    Converts seconds to a time in string format
    :param seconds: time in seconds
    :return: time in string format
    """
    hours = seconds // SECONDS_IN_HOUR
    minutes = (seconds % SECONDS_IN_HOUR) // SECONDS_IN_MINUTE
    seconds = seconds % SECONDS_IN_MINUTE

    return f"{hours}:{minutes}:{seconds}"
