"""Treat the engine like a product API: endpoints, tools, concurrency, cache.

Every case prints one line: PASS/FAIL, latency, and the per-request engine
counters the service already reports under the "ljqinfer" key.
"""
import json
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE, KEY = "http://127.0.0.1:8000", "devkey"
MODEL = "ljqinfer-dsv41f"
RESULTS = []


def http(path, body=None, key=KEY, method=None, raw=False, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 method=method or ("POST" if data else "GET"))
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("x-api-key", key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read().decode("utf-8", "replace")
            return r.status, (payload if raw else json.loads(payload)), time.time() - t0
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(payload)
        except ValueError:
            pass
        return e.code, payload, time.time() - t0


def stream(path, body, timeout=600):
    """Collect an SSE response: event order, text, and the final stats."""
    body = dict(body, stream=True)
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode())
    req.add_header("content-type", "application/json")
    req.add_header("x-api-key", KEY)
    events, text, stats, first = [], [], {}, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for rawline in r:
            line = rawline.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            ev = json.loads(line[6:])
            events.append(ev.get("type") or "chunk")
            if ev.get("ljqinfer"):
                stats = ev["ljqinfer"]
            delta = ev.get("delta") or {}
            piece = delta.get("text") or ""
            if not piece:
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.time() - t0
            text.append(piece)
    return events, "".join(text), stats, time.time() - t0, first


def msg(prompt, **kw):
    body = {"model": MODEL, "max_tokens": kw.pop("max_tokens", 160),
            "messages": [{"role": "user", "content": prompt}]}
    body.update(kw)
    return body


def blocks(reply, kind):
    return [b for b in (reply.get("content") or []) if b.get("type") == kind]


def text_of(reply):
    return "".join(b.get("text", "") for b in blocks(reply, "text"))


def record(name, ok, detail, seconds=None, stats=None):
    stats = stats or {}
    speed = ""
    if stats:
        speed = (" | prefill=%.2fs cache_hit=%s steps=%s step=%.1fms acc=%.2f"
                 % (stats.get("model_prefill_seconds") or 0,
                    stats.get("cache_hit_tokens", 0),
                    stats.get("decode_steps", "?"),
                    (stats.get("wall_ms_per_step") or 0),
                    stats.get("mtp_accepted_per_step") or 0))
    print("%-4s %-26s %6s  %s%s" % ("PASS" if ok else "FAIL", name,
                                    ("%.2fs" % seconds) if seconds else "",
                                    detail, speed), flush=True)
    RESULTS.append({"case": name, "ok": bool(ok), "detail": detail,
                    "seconds": seconds, "stats": stats})


# ---------------------------------------------------------------- endpoints
def case_endpoints():
    code, body, _ = http("/health")
    record("health", code == 200 and body.get("status") == "ok", str(body))

    code, body, _ = http("/v1/models")
    ids = [m.get("id") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    record("models", code == 200 and MODEL in ids, str(ids))

    code, body, _ = http("/v1/messages/count_tokens",
                         msg("count these tokens please"))
    n = body.get("input_tokens") if isinstance(body, dict) else None
    record("count_tokens", code == 200 and isinstance(n, int) and n > 0,
           "input_tokens=%s" % n)

    code, body, _ = http("/v1/messages", msg("hi"), key=None)
    record("auth_missing_key_401", code in (401, 403), "status=%s" % code)

    code, body, _ = http("/v1/messages", msg("hi"), key="wrong-key")
    record("auth_bad_key_401", code in (401, 403), "status=%s" % code)

    code, body, _ = http("/v1/messages", {"model": MODEL, "max_tokens": 8})
    record("bad_request_no_messages", code == 400, "status=%s %s" % (code, body))

    code, body, _ = http("/v1/messages",
                         {"model": MODEL, "max_tokens": -5,
                          "messages": [{"role": "user", "content": "hi"}]})
    record("bad_request_max_tokens", code == 400, "status=%s" % code)


# ------------------------------------------------------------------- chat
def case_chat():
    code, body, secs = http("/v1/messages", msg("用一句话解释什么是哈希表"))
    ok = code == 200 and body.get("stop_reason") == "end_turn" and text_of(body)
    record("chat_blocking", ok, repr(text_of(body)[:60]), secs, body.get("ljqinfer"))

    events, text, stats, secs, first = stream("/v1/messages",
                                              msg("用一句话解释什么是二分查找"))
    ok = ("message_start" in events and "message_stop" in events
          and len(text) > 5)
    record("chat_streaming", ok,
           "events=%d ttft=%.2fs %r" % (len(events), first or -1, text[:40]),
           secs, stats)

    body_multi = {"model": MODEL, "max_tokens": 80, "messages": [
        {"role": "user", "content": "记住这个数字：4173。回复 OK 即可。"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "我刚才让你记住的数字是多少？只回答数字。"}]}
    code, body, secs = http("/v1/messages", body_multi)
    out = text_of(body)
    record("chat_multi_turn", "4173" in out, repr(out[:60]), secs,
           body.get("ljqinfer"))

    code, body, secs = http("/v1/messages", msg(
        "介绍一下你自己", system="你是一只猫，每句话都必须以“喵”结尾。",
        max_tokens=80))
    out = text_of(body)
    record("chat_system_prompt", "喵" in out, repr(out[:60]), secs,
           body.get("ljqinfer"))

    code, body, secs = http("/v1/messages", msg(
        "从1数到20，用空格分隔", max_tokens=16))
    record("chat_max_tokens_stop", body.get("stop_reason") == "max_tokens",
           "stop=%s out=%s" % (body.get("stop_reason"),
                               (body.get("usage") or {}).get("output_tokens")),
           secs, body.get("ljqinfer"))

    code, body, secs = http("/v1/chat/completions", {
        "model": MODEL, "max_tokens": 60,
        "messages": [{"role": "user", "content": "Say exactly: pong"}]})
    out = ""
    if isinstance(body, dict) and body.get("choices"):
        out = (body["choices"][0].get("message") or {}).get("content") or ""
    record("openai_chat_completions", code == 200 and len(out) > 0,
           repr(out[:60]), secs, (body or {}).get("ljqinfer"))


# ------------------------------------------------------------ intelligence
QUIZ = [
    ("数学", "计算 (17 * 23) + 41 等于多少？只回答数字。", ["432"]),
    ("逻辑", "小明比小红高，小红比小刚高。三人中最矮的是谁？只回答名字。",
     ["小刚"]),
    ("常识", "水在标准大气压下的沸点是多少摄氏度？只回答数字。", ["100"]),
    ("代码", "Python 中 len([1,2,3,4]) 的结果是什么？只回答数字。", ["4"]),
    ("推理", "一个笼子里有鸡和兔共 10 只，腿共 28 条。兔子有几只？只回答数字。",
     ["4"]),
]


def case_quiz():
    for tag, q, answers in QUIZ:
        code, body, secs = http("/v1/messages", msg(q, max_tokens=200))
        out = text_of(body)
        ok = any(a in out for a in answers)
        record("quiz_" + tag, ok, repr(out.strip()[-40:]), secs,
               body.get("ljqinfer"))


# -------------------------------------------------------------- tool calls
WEATHER = {
    "name": "get_weather",
    "description": "查询某个城市的当前天气",
    "input_schema": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]},
}


def case_tools():
    body_req = {"model": MODEL, "max_tokens": 300, "tools": [WEATHER],
                "messages": [{"role": "user", "content": "北京现在天气怎么样？"}]}
    code, body, secs = http("/v1/messages", body_req)
    calls = blocks(body, "tool_use") if isinstance(body, dict) else []
    ok = code == 200 and calls and calls[0].get("name") == "get_weather"
    record("tool_use_emit", ok,
           "stop=%s call=%s" % (body.get("stop_reason"),
                                json.dumps(calls[:1], ensure_ascii=False)[:90]),
           secs, body.get("ljqinfer"))
    if not ok:
        record("tool_result_roundtrip", False, "skipped: no tool_use")
        return
    call = calls[0]
    follow = dict(body_req)
    follow["messages"] = body_req["messages"] + [
        {"role": "assistant", "content": body["content"]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call.get("id"),
            "content": "晴，气温 27 摄氏度，湿度 40%"}]}]
    code, body2, secs = http("/v1/messages", follow)
    out = text_of(body2)
    record("tool_result_roundtrip", "27" in out, repr(out[:70]), secs,
           body2.get("ljqinfer"))

    # tool choice must not fire when the question needs no tool
    code, body3, secs = http("/v1/messages", {
        "model": MODEL, "max_tokens": 120, "tools": [WEATHER],
        "messages": [{"role": "user", "content": "1 加 1 等于几？"}]})
    record("tool_not_called_when_useless", not blocks(body3, "tool_use"),
           repr(text_of(body3)[:50]), secs, body3.get("ljqinfer"))

    # OpenAI-style tools on the compat endpoint
    oai = {"model": MODEL, "max_tokens": 200, "tools": [{
        "type": "function", "function": {
            "name": "get_weather", "description": "查询天气",
            "parameters": WEATHER["input_schema"]}}],
        "messages": [{"role": "user", "content": "上海今天天气如何？"}]}
    code, body4, secs = http("/v1/chat/completions", oai)
    tc = []
    if isinstance(body4, dict) and body4.get("choices"):
        tc = (body4["choices"][0].get("message") or {}).get("tool_calls") or []
    record("openai_tool_calls", code == 200 and bool(tc),
           json.dumps(tc[:1], ensure_ascii=False)[:90], secs,
           (body4 or {}).get("ljqinfer"))


def case_thinking():
    code, body, secs = http("/v1/messages", msg(
        "一个数的平方是 169，这个数可能是多少？",
        thinking={"type": "enabled", "budget_tokens": 512}, max_tokens=600))
    think = blocks(body, "thinking")
    out = text_of(body)
    record("thinking_mode", code == 200 and bool(think),
           "think_chars=%d answer=%r" % (
               sum(len(b.get("thinking", "")) for b in think), out[:40]),
           secs, body.get("ljqinfer"))


# ------------------------------------------------------------- concurrency
def case_concurrency(n, label):
    prompt = "写一个 Python 函数计算斐波那契数列的第 n 项，并解释复杂度。"
    def one(i):
        t0 = time.time()
        code, body, secs = http("/v1/messages", msg(
            "[req%d] %s" % (i, prompt), max_tokens=128))
        st = body.get("ljqinfer") if isinstance(body, dict) else {}
        return {"i": i, "code": code, "secs": secs, "t0": t0,
                "stats": st or {}, "text": text_of(body) if isinstance(body, dict) else ""}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        out = list(ex.map(one, range(n)))
    wall = time.time() - t0
    ok = all(o["code"] == 200 and len(o["text"]) > 20 for o in out)
    steps = [o["stats"].get("wall_ms_per_step") or 0 for o in out]
    stalls = [o["stats"].get("prefill_stall_seconds") or 0 for o in out]
    lat = [o["secs"] for o in out]
    toks = sum((o["stats"].get("output_tokens") or 0) for o in out)
    record("concurrency_%s" % label, ok,
           "wall=%.2fs lat[min/max]=%.2f/%.2f step[min/max]=%.1f/%.1f "
           "stall_max=%.2fs tok/s=%.1f" % (
               wall, min(lat), max(lat), min(steps), max(steps),
               max(stalls), toks / wall if wall else 0))
    return out


# ------------------------------------------------------------------ cache
def case_cache():
    # the prefix cache survives across runs, so a fixed prompt is already warm
    # on the second run of this suite; a unique nonce guarantees a cold first
    # request and keeps the cold/warm comparison meaningful.
    nonce = "文档编号 %s。" % uuid.uuid4().hex
    long_prompt = ("以下是一份技术文档，请阅读后回答问题。" + nonce + "\n" +
                   ("缓存机制的设计要点包括：块对齐、前缀复用、淘汰策略。" * 260) +
                   "\n问题：上文反复提到的三个设计要点是什么？")
    code, b1, s1 = http("/v1/messages", msg(long_prompt, max_tokens=80))
    st1 = b1.get("ljqinfer") or {}
    code, b2, s2 = http("/v1/messages", msg(long_prompt, max_tokens=80))
    st2 = b2.get("ljqinfer") or {}
    u2 = b2.get("usage") or {}
    ok = (st2.get("cache_hit_tokens") or 0) > (st1.get("cache_hit_tokens") or 0)
    record("cache_warm_hit", ok,
           "input=%s cold: hit=%s prefill=%.2fs | warm: hit=%s prefill=%.2fs "
           "cache_read=%s | latency %.2fs -> %.2fs" % (
               st1.get("input_tokens"), st1.get("cache_hit_tokens"),
               st1.get("model_prefill_seconds") or 0,
               st2.get("cache_hit_tokens"),
               st2.get("model_prefill_seconds") or 0,
               u2.get("cache_read_input_tokens"), s1, s2))

    short = "用一句话说明什么是快速排序。"
    code, c1, t1 = http("/v1/messages", msg(short, max_tokens=64))
    code, c2, t2 = http("/v1/messages", msg(short, max_tokens=64))
    same = text_of(c1) == text_of(c2)
    record("determinism_repeat", same,
           "identical=%s %.2fs/%.2fs" % (same, t1, t2))


def main():
    print("== ljqinfer API acceptance ==", flush=True)
    case_endpoints()
    case_chat()
    case_quiz()
    case_tools()
    case_thinking()
    case_cache()
    case_concurrency(4, "b4")
    case_concurrency(8, "b8_queue")
    ok = sum(1 for r in RESULTS if r["ok"])
    print("\nTOTAL %d/%d passed" % (ok, len(RESULTS)), flush=True)
    for r in RESULTS:
        if not r["ok"]:
            print("  FAILED: %s -> %s" % (r["case"], r["detail"]), flush=True)
    with open("/tmp/api_suite.json", "w") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
