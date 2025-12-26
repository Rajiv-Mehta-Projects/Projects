import React from "react";

import { View, StyleSheet, TouchableOpacity } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import AppColors from "../config/AppColors";

function AppMenuIcon({ icon, onPress, ...otherProps }) {
  return (
    <TouchableOpacity onPress={onPress}>
      <View style={styles.button}>
        <MaterialCommunityIcons name={icon} size={55} />
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: "row",
    alignItems: "flex-start",
    backgroundColor: AppColors.white,
    justifyContent: "flex-start",
    borderRadius: 30,
  },
});

export default AppMenuIcon;
