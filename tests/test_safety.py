from heart_say.safety import assess_risk, crisis_response, detect_audience, guard_output, redact_pii


def test_suicide_is_critical():
    risk = assess_risk("我已经决定今晚跳楼了，你别劝我。")
    assert risk.level == "critical"
    assert risk.action == "protocol"
    assert "suicide" in risk.categories


def test_negation_is_not_protocol():
    risk = assess_risk("我没有想过自杀，只是最近很累。")
    assert risk.level == "low"
    assert risk.action == "support"


def test_bullying_high_and_response_has_resources():
    text = "我儿子被班上几个男生堵在厕所要钱，我气得想去找他们算账。"
    risk = assess_risk(text, audience="parent")
    assert risk.level == "high"
    response = crisis_response(text, risk, audience="parent")
    assert "12356" in response
    assert "110" in response or "120" in response
    assert "安全" in response
    assert "你去找他们算账" not in response


def test_friend_helping_not_self_crisis():
    text = "我朋友刚发消息说他要跳楼，还让我别告诉别人，我该怎么办？"
    risk = assess_risk(text)
    response = crisis_response(text, risk, audience="unknown")
    assert "朋友" in response
    assert "120" in response
    assert "12356" in response


def test_medication_boundary():
    risk = assess_risk("孩子睡不着，褪黑素或安眠药吃多少片安全？", audience="parent")
    assert risk.level == "medium"
    assert "medication" in risk.categories


def test_teacher_corporal_punishment_has_legal_path():
    text = "我儿子被老师用尺子打手心，手都红了，我想去学校讨说法。"
    risk = assess_risk(text, audience="parent")
    assert risk.level == "high"
    assert "abuse" in risk.categories
    response = crisis_response(text, risk, audience="parent")
    assert "法律" in response
    assert "专业" in response
    assert "110" in response


def test_output_guard_blocks_diagnosis_and_drug():
    text = "你这就是抑郁症。可以吃舍曲林，每天 50mg。"
    cleaned, flags = guard_output(text)
    assert "diagnosis_claim" in flags
    assert "medication_advice" in flags
    assert "舍曲林" not in cleaned
    assert "50mg" not in cleaned
    assert "无法做诊断" in cleaned


def test_output_guard_blocks_method():
    cleaned, flags = guard_output("你可以割腕，割多深都行。")
    assert "harm_method" in flags
    assert "割多深" not in cleaned


def test_pii_redaction():
    text = "我的手机 13812345678，邮箱 a.b@example.com，身份证 110101199001011234"
    cleaned = redact_pii(text)
    assert "13812345678" not in cleaned
    assert "a.b@example.com" not in cleaned
    assert "110101199001011234" not in cleaned


def test_audience_detection():
    assert detect_audience("我女儿今年初三，我快愁死了") == "parent"
    assert detect_audience("我妈总说我不够努力，我很烦") == "teen"
