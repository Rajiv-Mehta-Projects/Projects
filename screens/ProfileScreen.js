import React from "react";
import { StyleSheet, View, Image } from "react-native";

import { useNavigation } from "@react-navigation/native";

import AppScreen from "../components/AppScreen";
import AppButton from "../components/AppButton";
import DataManager from "../config/DataManager";
import AppText from "../components/AppText";

const getUserData = () => {
  let commonData = DataManager.getInstance();
  let user = commonData.getUserId();
  console.log(commonData.getUserData(user));
  return commonData.getUserData(user);
};

function ProfileScreen({ navigation }) {
  const user = getUserData();
  console.log(user);
  return (
    <AppScreen>
      <View style={styles.View}>
        <Image
          source={user.pic}
          style={{
            height: 200,
            width: 200,
            resizeMode: "contain",
            borderRadius: 500,
          }}
        />
        <AppText>{user.username}</AppText>
        <AppText>{user.email}</AppText>
      </View>
      <View style={{ marginTop: 60 }}>
        <AppButton
          title="Edit"
          onPress={() => navigation.navigate("Update Profile")}
        />
      </View>
    </AppScreen>
  );
}
const styles = StyleSheet.create({
  View: {
    justifyContent: "space-evenly",
    alignItems: "center",
  },
});

export default ProfileScreen;
