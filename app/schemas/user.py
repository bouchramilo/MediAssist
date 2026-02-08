# Schémas pour inscription / login
from typing import Optional
from pydantic import BaseModel, EmailStr, model_validator, Field
from datetime import datetime
from typing_extensions import Self


class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = "USER"
    password: str
    password_repeat: str

    @model_validator(mode='after')
    def check_passwords_match(self) -> Self:
        if self.password != self.password_repeat:
            raise ValueError('Passwords do not match')
        return self


class UserCreate(UserBase):
    email: EmailStr = Field(..., description="Email de l'utilisateur")
    username: str = Field(..., min_length=3, max_length=50, description="Nom d'utilisateur")
    password: str = Field(..., min_length=6, description="Mot de passe")
    password_repeat: str = Field(..., description="Confirmation du mot de passe")

    @model_validator(mode='after')
    def validate_required_fields(self) -> Self:
        if not self.email:
            raise ValueError('Email is required')
        if not self.username:
            raise ValueError('Username is required')
        if not self.password:
            raise ValueError('Password is required')
        if not self.password_repeat:
            raise ValueError('Password confirmation is required')
        return self


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6, description="Nouveau mot de passe")
    password_repeat: Optional[str] = Field(None, description="Confirmation du nouveau mot de passe")

    @model_validator(mode='after')
    def check_passwords_match_if_provided(self) -> Self:
        if self.password is not None or self.password_repeat is not None:
            if self.password != self.password_repeat:
                raise ValueError('Passwords do not match')
            if self.password is None:
                raise ValueError('Password is required when providing password confirmation')
            if self.password_repeat is None:
                raise ValueError('Password confirmation is required when providing new password')
        return self


class UserInDBBase(BaseModel):
    id: int
    email: EmailStr
    username: str
    is_active: bool
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# Additional properties to return via API
class User(UserInDBBase):
    pass


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    password_hash: str


# Schéma pour la connexion (login)
class UserLogin(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: str = Field(..., min_length=6, description="Mot de passe")

    @model_validator(mode='after')
    def validate_credentials(self) -> Self:
        if not self.email and not self.username:
            raise ValueError('Either email or username must be provided')
        return self


# Schéma pour la réponse de connexion (avec token)
class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


# Schéma pour changer le mot de passe
class UserChangePassword(BaseModel):
    current_password: str = Field(..., description="Mot de passe actuel")
    new_password: str = Field(..., min_length=6, description="Nouveau mot de passe")
    new_password_repeat: str = Field(..., description="Confirmation du nouveau mot de passe")

    @model_validator(mode='after')
    def validate_passwords(self) -> Self:
        if self.new_password != self.new_password_repeat:
            raise ValueError('New passwords do not match')
        if self.current_password == self.new_password:
            raise ValueError('New password must be different from current password')
        return self