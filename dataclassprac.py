from dataclasses import dataclass
from pydantic import BaseModel, Field


class Person:
    name: str
    age: int

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


person1 = Person(35, "30.3")
print(f"Person is {person1.name}, {person1.age}");       

class Student(BaseModel):
    name: str
    age: int


one = Student(name="35", age=25)
print(f"Student is {one.name}, {one.age}");    
    


class User(BaseModel):
    name: str = Field(description="Name of the User", max_length=10)
    age: int = Field(description="Age of the User", ge=18)
    email: str = Field(default_factory=lambda: "user@example.com", description="Email of the User")



user1 = User(name="Alice", age=19, email="alice1")

print(f"User is {user1.name}, {user1.age}, {user1.email}");


print(User.schema_json(indent=2))