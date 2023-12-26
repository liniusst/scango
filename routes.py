import requests
from flask import flash, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)

# import frontend.forms as forms
import frontend.models as models
from frontend.app import app, bcrypt, fastapi_url

login_manager = LoginManager(app)


@login_manager.user_loader
def load_user(user_id: int):
    try:
        user_data = requests.get(f"{fastapi_url}/api/v1/accounts/{user_id}")
        if user_data.status_code == 200:
            user_dict = user_data.json()
            user = models.User(id=user_dict["id"], is_active=user_dict["is_active"])
            return user
    except requests.exceptions.ConnectionError as error:
        app.logger.error(f"Connection error while fetching user data: {error}")
        return None
    except requests.exceptions.RequestException as error:
        app.logger.error(f"Request error while fetching user data: {error}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@login_manager.unauthorized_handler
def guest_callback():
    return redirect(url_for("log_in"))


@app.errorhandler(404)
def page_not_found(error):
    # logger.error(f"404 Error: {request.url}")
    return render_template("404.html"), 404


@app.route("/registration", methods=["GET", "POST"])
def reg_in():
    if current_user.is_authenticated:
        # logger.debug("Authenticated user redirected to index")
        return redirect(url_for("index"))

    form = forms.RegistrationForm()
    if form.validate_on_submit():
        pwd_hash = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user_data = {
            "email": form.email.data,
            "password": pwd_hash,
            "username": form.username.data,
        }
        try:
            response = requests.post(f"{fastapi_url}/api/v1/accounts", json=user_data)
            if response.status_code == 200:
                flash("Account created successfully!", "success")
                # logger.debug(f"New user {form.username.data} registered")
                return redirect(url_for("log_in"))
            else:
                flash("Failed to create account. Please try again.", "danger")
                # logger.error("Failed to create new user")
        except requests.exceptions.RequestException as error:
            flash("Failed to create account, server error.", "danger")
            # logger.error(f"Error creating account: {error}")

    return render_template("sign_in.html", form=form)


@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    try:
        response = requests.get(f"{fastapi_url}/api/v1/accounts/{current_user.id}")
        if response.status_code == 200:
            user = response.json()
    except requests.exceptions.RequestException as error:
        # logger.error(f"Error while fetching user account: {error}")
        user = None

    try:
        response = requests.get(f"{fastapi_url}/api/v1/games/stats/{current_user.id}")
        if response.status_code == 200:
            stats = response.json()
    except requests.exceptions.RequestException as error:
        # logger.error(f"Error while fetching game stats: {error}")
        stats = None

    try:
        response = requests.get(
            f"{fastapi_url}/api/v1/games/stats/lasts/{current_user.id}"
        )
        if response.status_code == 200:
            five = response.json()
    except requests.exceptions.RequestException as error:
        # logger.error(f"Error while fetching last game stats: {error}")
        five = None

    # logger.debug(f"User: {current_user.id} account page information loaded")
    return render_template(
        "account.html",
        account=user,
        stats=stats,
        last_five_games=five,
    )


@app.route("/login", methods=["GET", "POST"])
def log_in():
    form = forms.LoginForm()
    if form.validate_on_submit():
        try:
            response = requests.get(
                f"{fastapi_url}/api/v1/accounts/email/{form.email.data}"
            )

            if response.status_code == 200:
                user_data = response.json()
                user_id = user_data.get("id")
                user_password = user_data.get("password")
                user_status = user_data.get("is_active")
                if user_data and bcrypt.check_password_hash(
                    user_password, form.password.data
                ):
                    user = models.User(id=user_id, is_active=user_status)
                    login_user(user)
                    return redirect(url_for("index"))
                else:
                    # logger.debug("Login failed, bad credentials.")
                    flash("Login failed. Check email email and password", "danger")
            else:
                # logger.debug(f"User with email {form.email.data} not found.")
                flash(f"User with email: {form.email.data} not found.", "danger")

        except requests.exceptions.RequestException as error:
            flash("Login failed, server error.", "danger")
            # logger.error(f"Error while logging: {error}")

    return render_template("log_in.html", title="Login", form=form)


@app.route("/logoff")
def atsijungti():
    logout_user()
    return redirect(url_for("index"))
