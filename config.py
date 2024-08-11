import os

class Config:
    SECRET_KEY = 'c0281ad6c3e5bb6fb585e80dfe6ea6b9db49872078230ef1'  # Replace with your generated secret key
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), "database", "users.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
