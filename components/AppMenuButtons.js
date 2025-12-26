import React from "react";
import { StyleSheet, View, TouchableHighlight, Text } from "react-native";

import AppColors from "../config/AppColors";

function AppMenuButtons({ name, onPress, ...otherProps }) {
  return (
    <View>
      <TouchableHighlight
        onPress={onPress}
        style={styles.container}
        underlayColor={AppColors.primaryColor}
      >
        <Text style={styles.text}>{name}</Text>
      </TouchableHighlight>
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    width: "100%",
    alignSelf: "center",
    justifyContent: "center",
    backgroundColor: AppColors.white,
    color: AppColors.primaryColor,
  },
  text: {
    fontWeight: "bold",
    fontSize: 40,
    fontFamily: "sans-serif-medium",
  },
});
export default AppMenuButtons;
