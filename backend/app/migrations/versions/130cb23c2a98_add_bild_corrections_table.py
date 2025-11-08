"""add bild.corrections table"""

from alembic import op
import sqlalchemy as sa

# ---- Alembic identifiers ----
revision = "130cb23c2a98"
down_revision = "e5550fffd774"   # Baseline-Revision
branch_labels = None
depends_on = None

SCHEMA = "bild"
TABLE = "corrections"

def upgrade():
    # Schema sicherstellen (idempotent)
    op.execute('CREATE SCHEMA IF NOT EXISTS "bild";')

    op.create_table(
        TABLE,
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("published", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("article_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema=SCHEMA,
    )

    # Eindeutigkeit (gleiche Meldung nicht doppelt)
    op.create_unique_constraint(
        "uq_bild_corrections_unique",
        f"{SCHEMA}.{TABLE}",
        ["source_url", "published", "title"],
    )

    # Hilfsindizes
    op.create_index(
        "idx_bild_corrections_published",
        f"{SCHEMA}.{TABLE}",
        ["published"],
        unique=False,
    )
    op.create_index(
        "idx_bild_corrections_source_url",
        f"{SCHEMA}.{TABLE}",
        ["source_url"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "idx_bild_corrections_source_url",
        table_name=TABLE,
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_bild_corrections_published",
        table_name=TABLE,
        schema=SCHEMA,
    )
    op.drop_constraint(
        "uq_bild_corrections_unique",
        f"{SCHEMA}.{TABLE}",
        type_="unique",
    )
    op.drop_table(TABLE, schema=SCHEMA)
