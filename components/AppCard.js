import React from "react";
import { View, Image, Text, StyleSheet } from "react-native";


import AppColors from "../config/AppColors";

function AppCard({ title, image }) {
  return (
    <View style={styles.container}>
     {isFinite(image)? <Image source={image} style={styles.image} /> :<Image source={{uri: image}} style={styles.image}/>}
     <View>
      <Text style={{color: AppColors.black, backgroundColor:AppColors.white}} >
        {title}
      </Text>
      </View>
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    flexWrap: "wrap",
    width: "33%",
    justifyContent: "center",
    marginRight: 3,
    overflow:"hidden",
    backgroundColor: AppColors.white,
    padding: 10
  },
  image: {
    height: 100,
    width: 100,
    justifyContent: "space-around",
  },
});

export default AppCard;
