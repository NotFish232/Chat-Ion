if user_agent and not any(
            (
                any(user_agent_substring in user_agent for user_agent_substring in settings.NONLOGGABLE_USER_AGENT_SUBSTRINGS),
                any(map(path.startswith,settings.NONLOGGABLE_PATH_BEGINNINGS)),
                any(path.endswith(settings.NONLOGGABLE_PATH_ENDINGS))),
            )
        ):