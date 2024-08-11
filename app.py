import pickle
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session
from models import db, User
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor as GP
from deap import base, creator, tools, gp
from gplearn.genetic import SymbolicRegressor as GEP

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)

    with app.app_context():
        db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "database", "users.db")
        if not os.path.exists(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db.create_all()

    @app.route('/')
    def home():
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            return render_template('home.html', user=user)
        return redirect(url_for('login'))

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('home'))
        return render_template('signup.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()
            if user and user.password == password:
                session['user_id'] = user.id
                return redirect(url_for('upload'))
            else:
                return 'Invalid credentials'
        return render_template('login.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                file_path = os.path.join('uploads', 'dataset.csv')
                file.save(file_path)
                return redirect(url_for('model_selection'))
        return render_template('upload.html')

    @app.route('/model_selection', methods=['GET', 'POST'])
    def model_selection():
        if request.method == 'POST':
            #param = int(request.form['param'])
            model_type = request.form.get('model', 'gp')  # Default to GP if no model type is selected

            file_path = 'uploads/dataset.csv'
            df = pd.read_csv(file_path)
            X = df[['x0', 'x1']].values
            y = df['y'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            if model_type == 'gp':
                est = GP(population_size=5000,
                         generations=30, stopping_criteria=0.01,
                         p_crossover=0.7, p_subtree_mutation=0.1,
                         p_hoist_mutation=0.05, p_point_mutation=0.1,
                         max_samples=0.9, verbose=1,
                         parsimony_coefficient=0.01, random_state=0)
                start_time = time.time()
                est.fit(X_train, y_train)
                elapsed_time = time.time() - start_time

                # Save the GP model
                model_path = 'models/gp_model.pkl'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(est, f)

                return redirect(url_for('results', model_type='gp', time_taken=elapsed_time))

            elif model_type == 'gep':
                # Setup GEP
                # Replace with actual GEP implementation
                est = GEP(population_size=5000,
                          generations=30, stopping_criteria=0.01,
                          p_crossover=0.7, p_subtree_mutation=0.1,
                          p_hoist_mutation=0.05, p_point_mutation=0.1,
                          max_samples=0.9, verbose=1,
                          parsimony_coefficient=0.01, random_state=0)
                start_time = time.time()
                est.fit(X_train, y_train)
                elapsed_time = time.time() - start_time

                # Save the GEP model
                model_path = 'models/gep_model.pkl'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(est, f)

                return redirect(url_for('results', model_type='gep', time_taken=elapsed_time))

        return render_template('model_selection.html')

    @app.route('/results')
    def results():
        model_type = request.args.get('model_type', 'gp')
        time_taken = float(request.args.get('time_taken', 0))
        
        # Round time_taken to 2 decimal places
        time_taken = round(time_taken, 2)
        
        if model_type == 'gp':
            model_path = 'models/gp_model.pkl'
        elif model_type == 'gep':
            model_path = 'models/gep_model.pkl'
        else:
            return redirect(url_for('model_selection'))

        if not os.path.exists(model_path):
            return redirect(url_for('model_selection'))

        with open(model_path, 'rb') as f:
            est = pickle.load(f)

        file_path = 'uploads/dataset.csv'
        df = pd.read_csv(file_path)
        X = df[['x0', 'x1']].values
        y_truth = df['y'].values

        x0 = np.arange(-1, 1.1, 0.1)
        x1 = np.arange(-1, 1.1, 0.1)
        x0, x1 = np.meshgrid(x0, x1)

        # Ensure the output of the model is reshaped correctly
        y_pred = est.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
        score = est.score(X, y_truth)

        # Round score to 6 decimal places
        score = round(score, 6)

        fig = plt.figure(figsize=(12, 10))

        for i, (y, title) in enumerate([(y_truth, "Ground Truth"),
                                        (y_pred, "Model Prediction")]):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks(np.arange(-1, 1.1, 0.5))
            ax.set_yticks(np.arange(-1, 1.1, 0.5))

            # Plot the surface
            surf = ax.plot_surface(x0, x1, y.reshape(x0.shape), rstride=1, cstride=1, color='blue', alpha=0.4)
            points = ax.scatter(X[:, 0], X[:, 1], y_truth, color='black')

            if title == "Model Prediction" and score is not None:
                ax.text(-0.7, 0.1, 0.1, "$R^2 = %.6f$" % score, fontsize=14)
            plt.title(title)

        plt.show()

        return render_template('results.html', equation=str(est._program), score=score, time_taken=time_taken)


    @app.route('/settings', methods=['GET', 'POST'])
    def settings():
        if request.method == 'POST':
            theme = request.form.get('theme', 'light')
            session['theme'] = theme
            return redirect(url_for('settings'))
        return render_template('settings.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
