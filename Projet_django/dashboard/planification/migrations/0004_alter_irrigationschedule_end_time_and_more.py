# Generated by Django 4.2 on 2024-10-27 12:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('planification', '0003_irrigationschedule_irrigationplan_schedule'),
    ]

    operations = [
        migrations.AlterField(
            model_name='irrigationschedule',
            name='end_time',
            field=models.DateTimeField(),
        ),
        migrations.AlterField(
            model_name='irrigationschedule',
            name='start_time',
            field=models.DateTimeField(),
        ),
    ]