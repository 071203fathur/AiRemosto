"""empty message

Revision ID: d688fdd864c4
Revises: 5af429264c38
Create Date: 2023-05-26 15:38:05.807225

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd688fdd864c4'
down_revision = '5af429264c38'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('age',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.Integer(), nullable=False),
    sa.Column('anak', sa.Integer(), nullable=True),
    sa.Column('remaja', sa.Integer(), nullable=True),
    sa.Column('dewasa', sa.Integer(), nullable=True),
    sa.Column('lansia', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('age', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_age_anak'), ['anak'], unique=False)
        batch_op.create_index(batch_op.f('ix_age_dewasa'), ['dewasa'], unique=False)
        batch_op.create_index(batch_op.f('ix_age_lansia'), ['lansia'], unique=False)
        batch_op.create_index(batch_op.f('ix_age_remaja'), ['remaja'], unique=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('age', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_age_remaja'))
        batch_op.drop_index(batch_op.f('ix_age_lansia'))
        batch_op.drop_index(batch_op.f('ix_age_dewasa'))
        batch_op.drop_index(batch_op.f('ix_age_anak'))

    op.drop_table('age')
    # ### end Alembic commands ###
