from src.agent.rewrite import is_follow_up_question


def test_short_independent_question_is_not_follow_up():
    history = [{"role": "user", "content": "Làm thế nào để push project lên GitHub bằng SSH?"}]
    assert is_follow_up_question("passkey là gì", history) is False


def test_https_question_is_follow_up():
    history = [{"role": "user", "content": "Làm thế nào để push project lên GitHub bằng SSH?"}]
    assert is_follow_up_question("Còn HTTPS thì sao?", history) is True