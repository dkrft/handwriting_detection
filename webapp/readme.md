
# running

```bash
cd webapp/hwdetect_webapp
python manage.py runserver
```

# login

http://127.0.0.1:8000/accounts/login/

- proxima
- admin

was created using:

```bash
python manage.py createsuperuser
```

# some other commands

create new directory for an app/subpage:
```bash
django-admin startapp upload
```

I'm not exactly sure what this is. It dumps
some files into the static root:
```bash
python manage.py collectstatic
```

# working

- https://www.bogotobogo.com/python/Django/Python_Django_hello_world.php
- https://wsvincent.com/django-user-authentication-tutorial-login-and-logout/
- https://docs.djangoproject.com/en/2.1/topics/templates/

# not working or outdated

because some things were just different after the initial installation for me
- https://djangoforbeginners.com/hello-world/
- https://dfpp.readthedocs.io/en/latest/chapter_01.html
