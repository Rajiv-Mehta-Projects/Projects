import React from "react";

import { createStackNavigator } from "@react-navigation/stack";

import WelcomeScreen from "../screens/WelcomeScreen";
import RegisterScreen from "../screens/RegisterScreen";
import MainScreen from "../screens/MainScreen";
import MenuScreen from "../screens/MenuScreen";
import ProfileScreen from "../screens/ProfileScreen";
import NewMemoryScreen from "../screens/NewMemoryScreen";
import UpdateProfileScreen from "../screens/UpdateProfileScreen";

const AppStack = createStackNavigator();

const AuthNavigator = () => (
  <AppStack.Navigator>
    <AppStack.Screen
      name="Welcome"
      component={WelcomeScreen}
      options={{ headerShown: false }}
    />
    <AppStack.Screen
      name="Register"
      component={RegisterScreen}
      options={{ title: "RegisterScreen" }}
    />
    <AppStack.Screen
      name="Main"
      component={MainScreen}
      options={{ headerShown: false }}
    />
    <AppStack.Screen
      name="Menu"
      component={MenuScreen}
      options={{ headerShown: false }}
    />
    <AppStack.Screen
      name="Profile"
      component={ProfileScreen}
      options={{ headerShown: true }}
    />
    <AppStack.Screen
      name="NewMemory"
      component={NewMemoryScreen}
      options={{ headerShown: false }}
    />
    <AppStack.Screen
      name="Update Profile"
      component={UpdateProfileScreen}
      options={{ headerShown: true }}
    />
  </AppStack.Navigator>
);

export default AuthNavigator;
