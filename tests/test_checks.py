from manny.checks import check_report, check_web

# What the system produced for a bakery landing page before these checks existed.
BAD_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corsaro Bread</title><link rel="stylesheet" href="styles.css"></head>
<body><header><h1>Welcome to Corsaro Bread</h1></header>
<main><div class="product"><img src="path/to/image1.jpg" alt="Bread Product 1">
<h3>Bread Product 1</h3><p>Description of Bread Product 1.</p></div>
<!-- Add more products as needed --></main>
<script src="script.js"></script></body></html>"""
BAD_CSS = "body { font-family: Arial; } h1 { color: #333; } p { color: #666; }"

GOOD_TEXT = "Pane a lievitazione naturale sfornato ogni mattina nel nostro forno a legna. " * 8
GOOD_PAGE = f"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forno Corsaro</title><link rel="stylesheet" href="css/style.css"></head>
<body><h1>Forno Corsaro</h1><p>{GOOD_TEXT}</p>
<a href="#contatti">Contatti</a><a href="https://maps.example.org/forno">Mappa</a>
<a href="mailto:info@fornocorsaro.it">Scrivici</a><img src="pane.svg" alt="Pagnotta"></body></html>"""
GOOD_CSS = "\n".join(f".block-{i} {{ margin: {i}px; }}" for i in range(10))


def write(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def test_the_original_landing_page_is_rejected_for_the_right_reasons(tmp_path):
    write(tmp_path, {"index.html": BAD_PAGE, "styles.css": BAD_CSS, "script.js": "// Add any JavaScript here"})
    passed, report = check_web(tmp_path)
    assert not passed
    assert "links to 'path/to/image1.jpg', which does not exist" in report
    assert "placeholder text" in report and "Product 1" in report and "delete those lines" in report
    assert "almost no text content" in report
    assert "only 3 CSS rules" in report
    assert "script.js: placeholder text ['Add any JavaScript here']" in report


def test_a_complete_page_passes(tmp_path):
    write(tmp_path, {"index.html": GOOD_PAGE, "css/style.css": GOOD_CSS, "pane.svg": "<svg/>"})
    passed, report = check_web(tmp_path)
    assert passed, report


def test_page_structure_problems(tmp_path):
    write(tmp_path, {"index.html": f"<html><body><p>{GOOD_TEXT}</p><style>{GOOD_CSS}</style></body></html>"})
    passed, report = check_web(tmp_path)
    assert not passed and "no <title>" in report and "no viewport" in report
    assert "CSS rules" not in report


def test_no_page_at_all(tmp_path):
    assert check_web(tmp_path) == (False, "no .html file was found in the workspace")


SEEN = {"https://ollama.com/library/qwen2.5-coder", "https://huggingface.co/blog/models/"}
BODY = "Qwen2.5-Coder è una famiglia di modelli per il codice con varianti da 0.5 a 32 miliardi di parametri. " * 6


def test_report_grounded_in_consulted_sources_passes(tmp_path):
    write(tmp_path, {"report.md": f"{BODY}\nFonte: https://ollama.com/library/qwen2.5-coder. Vedi (https://huggingface.co/blog/models)."})
    passed, report = check_report(tmp_path, SEEN)
    assert passed and "2 source(s)" in report


def test_report_citing_urls_never_seen_is_rejected(tmp_path):
    write(tmp_path, {"report.md": f"{BODY}\nFonte: https://invented.example.org/benchmarks"})
    passed, report = check_report(tmp_path, SEEN)
    assert not passed and "https://invented.example.org/benchmarks" in report


def test_report_must_exist_have_substance_and_sources(tmp_path):
    assert check_report(tmp_path, SEEN) == (False, "write the final answer to report.md")
    write(tmp_path, {"report.md": "1. Qwen 3.5 7B. Alibaba\n2. Llama 4 Scout. Meta"})
    passed, report = check_report(tmp_path, SEEN)
    assert not passed and "too short" in report and "cites no source URL" in report


def test_placeholder_image_services_are_flagged(tmp_path):
    page = GOOD_PAGE.replace('src="pane.svg"', 'src="https://via.placeholder.com/150"')
    write(tmp_path, {"index.html": page, "css/style.css": GOOD_CSS})
    passed, report = check_web(tmp_path)
    assert not passed and "placeholder.com" in report
