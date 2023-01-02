# Denial-of-Service Protection
___________________________________

Denial of service attacks are critical attacks. In BastionLab, the simplest attacks would be to use up all available memory by creating several dataframes or to use up all available storage by saving several dataframes.

We defend against such attacks by tracking users' actions. If a user calls collect() to run queries several times in under ten seconds, they will be temporarily blocked.

The number of queries that can be run in ten seconds before the server blocks the user is set by the data owner. The max_consecutive_runs parameter in the config.toml is this value. By default, the value is 10.

Similarly, the number of saves that can be performed in twenty seconds is set by the owner in the config.toml, this is the max_consecutive_saves variable.
Should a user perform more saves than the data owner has approved within twenty seconds, they will be temporarily blocked.

The amount of time for which a user is blocked is also set by the data owner in the config.toml. The user_ban_time variable is this value, it is set in seconds and by default it is 900 seconds or 15 minutes.