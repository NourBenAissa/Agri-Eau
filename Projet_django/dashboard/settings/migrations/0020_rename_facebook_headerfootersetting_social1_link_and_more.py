# Generated by Django 4.2.6 on 2023-10-29 13:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('settings', '0019_websitesetting_twilio_auth_token_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='headerfootersetting',
            old_name='facebook',
            new_name='social1_link',
        ),
        migrations.RenameField(
            model_name='headerfootersetting',
            old_name='instagram',
            new_name='social2_link',
        ),
        migrations.RenameField(
            model_name='headerfootersetting',
            old_name='linkedin',
            new_name='social3_link',
        ),
        migrations.RemoveField(
            model_name='headerfootersetting',
            name='twitter',
        ),
        migrations.AddField(
            model_name='headerfootersetting',
            name='social1_icon',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='headerfootersetting',
            name='social2_icon',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='headerfootersetting',
            name='social3_icon',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='headerfootersetting',
            name='social4_icon',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='headerfootersetting',
            name='social4_link',
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
        migrations.AddField(
            model_name='seosetting',
            name='tag_line',
            field=models.CharField(blank=True, max_length=600, null=True),
        ),
    ]