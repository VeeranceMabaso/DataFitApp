from models import db, User
from config import Config
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app

def view_users():
    app = create_app()
    with app.app_context():
        # Query all users
        users = User.query.all()

        # Print user information
        for user in users:
            print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}, Password: {user.password}")

if __name__ == '__main__':
    view_users()
