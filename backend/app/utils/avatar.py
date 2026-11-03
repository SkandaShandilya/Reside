from app.models.user import User


def avatar_url_for(user: User) -> str | None:
    """Relative API URL of the user's profile picture (None if unset).
    `v` busts the browser cache when the picture is replaced."""
    if not user.avatar_path:
        return None
    version = int(user.updated_at.timestamp()) if user.updated_at else 0
    return f"/api/profile/avatar/{user.id}?v={version}"
