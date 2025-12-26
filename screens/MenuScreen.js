import React from "react";
import { View, StyleSheet} from "react-native";

import AppScreen from "../components/AppScreen";
import AppMenuButtons from "../components/AppMenuButtons";

function MenuScreen({ navigation }) {
  return (
    <AppScreen>
      <View style={styles.container}>
        <AppMenuButtons
          name="Profile"
          onPress={() => navigation.navigate("Profile")}
        />
        <AppMenuButtons
          name="Memories"
          onPress={() => navigation.navigate("Main")}
        />
        <AppMenuButtons
          name="Logout"
          onPress={() => navigation.navigate("Welcome")}
        />
      </View>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "space-evenly",
  },
});

export default MenuScreen;
