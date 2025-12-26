import React from "react";
import {  StyleSheet, TextInput,View } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";


import AppColors from "../config/AppColors";

function AppTextInput({ icon, ...otherProps }) {
  return (
    <View style={styles.container}>
      <MaterialCommunityIcons name={icon} size={22} />
      <TextInput style={styles.textInput} {...otherProps} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
    backgroundColor: AppColors.white,
    flexDirection: "row",
    borderRadius: 25,
    padding: 5,
    marginVertical: 15,
    width: "75%",
  },
  textInput: {
    flex: 1,
    fontSize: 15,
    borderRadius: 15,
    width: "50%",
    marginVertical: 10,
    marginLeft: 10,
  },
});

export default AppTextInput;
