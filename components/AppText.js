import React from "react";
import { StyleSheet,Text} from "react-native";

import AppColors from "../config/AppColors";

function AppText({ children }) {
  return <Text style={styles.text}> {children} </Text>;
}

const styles = StyleSheet.create({
  text: {
    fontSize: 40,
    fontFamily: "sans-serif-medium",
    color: AppColors.primaryColor,
    alignSelf: "center",
    fontWeight: "bold",
  },
});
export default AppText;
