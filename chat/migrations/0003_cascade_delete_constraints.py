# Custom migration to enforce ON DELETE CASCADE at the PostgreSQL database schema level
# This allows direct deletion of conversations or users in the Neon console / SQL editor
# without triggering Foreign Key violation errors.

from django.db import migrations

def apply_cascade_constraints(apps, schema_editor):
    if schema_editor.connection.vendor == 'postgresql':
        with schema_editor.connection.cursor() as cursor:
            # 1. First clean up any pre-auth orphaned test conversations (user_id IS NULL)
            cursor.execute("DELETE FROM chat_message WHERE conversation_id IN (SELECT id FROM chat_conversation WHERE user_id IS NULL);")
            cursor.execute("DELETE FROM chat_conversation WHERE user_id IS NULL;")

            # 2. Add ON DELETE CASCADE to chat_message -> chat_conversation
            cursor.execute("""
                ALTER TABLE chat_message
                DROP CONSTRAINT IF EXISTS chat_message_conversation_id_a1207bf4_fk_chat_conversation_id;
                
                ALTER TABLE chat_message
                DROP CONSTRAINT IF EXISTS chat_message_conversation_id_fk_cascade;

                ALTER TABLE chat_message
                ADD CONSTRAINT chat_message_conversation_id_fk_cascade
                FOREIGN KEY (conversation_id)
                REFERENCES chat_conversation(id)
                ON DELETE CASCADE;
            """)

            # 3. Add ON DELETE CASCADE to chat_conversation -> auth_user
            cursor.execute("""
                ALTER TABLE chat_conversation
                DROP CONSTRAINT IF EXISTS chat_conversation_user_id_15357b84_fk_auth_user_id;

                ALTER TABLE chat_conversation
                DROP CONSTRAINT IF EXISTS chat_conversation_user_id_fk_cascade;

                ALTER TABLE chat_conversation
                ADD CONSTRAINT chat_conversation_user_id_fk_cascade
                FOREIGN KEY (user_id)
                REFERENCES auth_user(id)
                ON DELETE CASCADE;
            """)

            # 4. Add ON DELETE CASCADE to authtoken_token -> auth_user
            cursor.execute("""
                ALTER TABLE authtoken_token
                DROP CONSTRAINT IF EXISTS authtoken_token_user_id_35299eff_fk_auth_user_id;

                ALTER TABLE authtoken_token
                DROP CONSTRAINT IF EXISTS authtoken_token_user_id_fk_cascade;

                ALTER TABLE authtoken_token
                ADD CONSTRAINT authtoken_token_user_id_fk_cascade
                FOREIGN KEY (user_id)
                REFERENCES auth_user(id)
                ON DELETE CASCADE;
            """)

def rollback_cascade_constraints(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    atomic = False

    dependencies = [
        ('chat', '0002_conversation_user'),
    ]

    operations = [
        migrations.RunPython(apply_cascade_constraints, rollback_cascade_constraints),
    ]
