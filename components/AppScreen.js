import React from "react";
import { SafeAreaView, StyleSheet } from "react-native";
import Constants from "expo-constants";


import AppColors from "../config/AppColors";

function AppScreen({ children, style }) {
  return <SafeAreaView style={[styles.image, style]}>{children}</SafeAreaView>;
}

const styles = StyleSheet.create({
  image: {
    flex: 1,
    marginTop: Constants.statusBarHeight,
    backgroundColor: AppColors.backColor,
  },
});

export default AppScreen;
