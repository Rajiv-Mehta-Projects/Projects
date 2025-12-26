export default class DataManager {
  // Arrays + Getters and Setters for users, memories and categories

  static myInstance = null;
  userID = "";
  users = [
    {
      id: "user1",
      username: "Obi-wan",
      email: "N@gmail.com",
      password: "1234",
      pic: require("../assets/ObiWan.jpg"),
    },
    {
      id: "user2",
      username: "Anakin",
      email: "A@gmail.com",
      password: "2345",
      pic: require("../assets/Anakin.jpeg"),
    },
  ];
  memories = [
    {
      userId: "user1",
      title: "Space",
      category: "Science",
      image: require("../assets/Background.jpg"),
    },
    {
      userId: "user2",
      title: "One Punch Man",
      category: "Science",
      image: require("../assets/onePunch.jpg"),
    },
    {
      userId: "user1",
      title: "Space2",
      category: "Science",
      image: require("../assets/Background.jpg"),
    },
  ];
  categories = [
    {
      label: "Adventure",
      value: 1,
    },
    {
      label: "Beach",
      value: 2,
    },
    {
      label: "Entertainment",
      value: 3,
    },
    {
      label: "Food",
      value: 4,
    },
    {
      label: "Holiday",
      value: 5,
    },
    {
      label: "Memes",
      value: 6,
    },
    {
      label: "Miscellaneous",
      value: 7,
    },
    {
      label: "Science",
      value: 8,
    },
  ];

  static getInstance() {
    if (DataManager.myInstance == null) {
      DataManager.myInstance = new DataManager();
    }
    return this.myInstance;
  }
  getUserId() {
    return this.userId;
  }
  getAllUsers() {
    return this.users;
  }
  getUser({ email }) {
    return this.users.find((user) => user.email === email);
  }
  getUserData(id) {
    return this.users.find((user) => user.id === id);
  }
  setUserId(id) {
    this.userId = id;
  }
  addUser(user) {
    this.users.push(user);
  }

  validateUser({ email, password }) {
    return (
      this.users.filter(
        (user) => user.email === email && user.password === password
      ).length > 0
    );
  }

  getMemory(id) {
    return this.memories.filter((memory) => memory.userId === id);
  }
  getMemories() {
    return this.memories;
  }

  addMemory(memory) {
    this.memories.push(memory);
  }

  getCategory() {
    return this.categories;
  }
}
