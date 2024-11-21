# Generated by Django 4.2.4 on 2023-08-31 14:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('about', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='aboutPage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('subtitle', models.CharField(max_length=100)),
                ('title', models.CharField(max_length=300)),
                ('description', models.TextField()),
                ('button1_text', models.CharField(max_length=100)),
                ('button1_url', models.CharField(max_length=500)),
                ('button2_text', models.CharField(max_length=100)),
                ('button2_url', models.CharField(max_length=500)),
                ('button3_text', models.CharField(max_length=100)),
                ('button3_url', models.CharField(max_length=500)),
                ('years_of_experience', models.IntegerField()),
                ('image1', models.ImageField(upload_to='AboutPage/')),
                ('image2', models.ImageField(upload_to='AboutPage/')),
                ('image3', models.ImageField(upload_to='AboutPage/')),
            ],
        ),
    ]