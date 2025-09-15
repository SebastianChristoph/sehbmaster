import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import psycopg

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://appuser:appsecret@db:5432/appdb")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("<html><body><h3>OK</h3><p>Siehe <a href=\"/test-names\">/test-names</a></p></body></html>")

@app.get("/test-names", response_class=HTMLResponse)
def list_test_names(request: Request):
    # frische Verbindung je Request (simpel & robust für dieses Minimalbeispiel)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT vorname, nachname FROM test."test-names" ORDER BY nachname, vorname;')
            rows = cur.fetchall()

    # rows: List[Tuple[vorname, nachname]]
    return templates.TemplateResponse(
        "test_names.html",
        {"request": request, "rows": rows}
    )
