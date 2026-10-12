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


from manny.checks import check_project_pages, wiring_problems

# The situation from a real run: a login page that looks complete and cannot work.
SERVER = '''from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register(): ...

@app.route('/login', methods=['POST'])
def login(): ...

@app.route('/user/<int:user_id>', methods=['GET'])
def get_user(user_id): ...
'''
LOGIN_PAGE = '''<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Accesso</title></head>
<body><form onsubmit="go(event)"><input id="u"><button>Accedi</button></form>
<script>
async function go(e) { e.preventDefault(); await fetch('/login', { method: 'POST' }); }
async function load() { await fetch(`/user/7`); }
</script></body></html>'''


def test_page_calling_a_server_that_never_serves_it_is_caught(tmp_path):
    write(tmp_path, {"server.py": SERVER, "index.html": LOGIN_PAGE})
    problems = wiring_problems(tmp_path)
    assert len(problems) == 1
    assert "relative URLs such as '/login'" in problems[0] and "Add a route for '/'" in problems[0]
    passed, report = check_project_pages(tmp_path)
    assert not passed and "never serves index.html" in report


def test_page_served_by_the_server_and_calling_real_routes_passes(tmp_path):
    served = SERVER + "\n@app.route('/')\ndef home():\n    return send_from_directory('.', 'index.html')\n"
    write(tmp_path, {"server.py": served, "index.html": LOGIN_PAGE})
    assert wiring_problems(tmp_path) == []
    # A login form has little text and little CSS: that is fine for a page that comes with a program.
    assert check_project_pages(tmp_path) == (True, "")


def test_call_to_a_route_that_does_not_exist_is_caught(tmp_path):
    page = LOGIN_PAGE.replace("fetch('/login'", "fetch('/api/signin'")
    served = SERVER + "\n@app.route('/')\ndef home(): ...\n"
    write(tmp_path, {"server.py": served, "static/app.js": "fetch('/logout')", "index.html": page.replace("<script>", '<script src="static/app.js"></script><script>')})
    problems = wiring_problems(tmp_path)
    assert any("calls '/api/signin', but the server defines no such route" in p for p in problems)
    assert any("calls '/logout'" in p for p in problems) and not any("/user/7" in p for p in problems)


def test_absolute_local_calls_need_the_page_served_or_cors(tmp_path):
    page = LOGIN_PAGE.replace("fetch('/login'", "fetch('http://localhost:5000/login'").replace("fetch(`/user/7`)", "1")
    write(tmp_path, {"server.py": SERVER, "index.html": page})
    assert any("CORS" in p for p in wiring_problems(tmp_path))
    write(tmp_path, {"server.py": SERVER + "\nfrom flask_cors import CORS\nCORS(app)\n"})
    assert wiring_problems(tmp_path) == []


def test_static_sites_and_form_actions(tmp_path):
    write(tmp_path, {"index.html": GOOD_PAGE, "css/style.css": GOOD_CSS, "pane.svg": "<svg/>"})
    assert wiring_problems(tmp_path) == []
    form = LOGIN_PAGE.replace('<form onsubmit="go(event)">', '<form action="/subscribe" method="post">')
    write(tmp_path / "app", {"server.py": SERVER + "\n@app.route('/')\ndef home(): ...\n", "index.html": form})
    assert any("calls '/subscribe'" in p for p in wiring_problems(tmp_path / "app"))
