# Generated by Django 4.2.4 on 2023-09-03 18:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('settings', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='websitesetting',
            old_name='messenger_is_active',
            new_name='facebook_chat_is_active',
        ),
    ]