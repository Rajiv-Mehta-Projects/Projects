import React from "react";
import { TouchableOpacity, StyleSheet, Text } from "react-native";

import AppScreen from "./AppScreen";
// Style for Categories drop down menu
function AppClickerItem({ onPress, label }) {
  return (
    <AppScreen>
      <TouchableOpacity style={styles.container} onPress={onPress}>
        <Text style={styles.text}> {label} </Text>
      </TouchableOpacity>
    </AppScreen>
  );
}
const styles = StyleSheet.create({
  container: {
    padding: 20,
    alignItems: "center",
  },
  text: {
    fontSize: 30,
    fontWeight: "bold",
  },
});

export default AppClickerItem;
