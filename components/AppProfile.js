import React from "react";
import { StyleSheet, View, Text } from "react-native";
import AppColors from "../config/AppColors";

function AppProfile({ username, email,}) {
  return (
    <View>
      <Text styles = {styles.text}> {username}</Text>
      <Text> {email} </Text>
    </View>
  );
}
const styles = StyleSheet.create({
  text: {
    flex:1,
    backgroundColor: AppColors.white,
  },
  icon: {
    resizeMode: "contain",
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: "flex-start",
    alignSelf: "center",
  },
});
export default AppProfile;
