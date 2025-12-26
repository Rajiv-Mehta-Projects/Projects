import React, { useEffect, useState } from "react";
import { StyleSheet, View, Image, FlatList } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import AppCard from "../components/AppCard";
import AppColors from "../config/AppColors";
import AppMenuIcon from "../components/AppMenuIcon";
import AppScreen from "../components/AppScreen";
import AppText from "../components/AppText";
import DataManager from "../config/DataManager";
import { useIsFocused } from "@react-navigation/native";

const getMemory = () => {
  let commonData = DataManager.getInstance();
  let user = commonData.getUserId();
  return commonData.getMemory(user);
};
// Sets a variable for number of columns of memories
const numCols = 3;

function MainScreen({ navigation }) {
  const memory = getMemory();
  const [memories, setMemories] = useState(memory);
  console.log("Main Screen Memories" + memories);
  const Focused = useIsFocused();
  useEffect(() => {
    setMemories(getMemory());
  }, [Focused]);
  return (
    <AppScreen>
      <View style={{ marginTop: -10 }}>
        <Image
          style={styles.welcomeLogo}
          source={require("../assets/Logo.png")}
        />
      </View>

      <View>
        <AppText>Memories</AppText>
      </View>
      <View
        style={{
          alignSelf: "center",
          backgroundColor: "white",
          borderRadius: 40,
          padding: 20,
        }}
      >
        <MaterialCommunityIcons
          name="plus-thick"
          size={50}
          onPress={() => navigation.navigate("NewMemory")}
          color={AppColors.primaryColor}
        />
      </View>

      <View style={styles.MenuButton}>
        <AppMenuIcon icon="menu" onPress={() => navigation.navigate("Menu")} />
      </View>
      <View style={styles.container}>
        <FlatList
          data={memories}
          keyExtractor={(memories) => memories.userId.toString()}
          numColumns={numCols}
          refreshing={true}
          onRefresh={() => memories}
          contentContainerStyle={{ paddingBottom: 200 }}
          renderItem={({ item }) => (
            <AppCard title={item.title} image={item.image} />
          )}
        ></FlatList>
      </View>
    </AppScreen>
  );
}
const styles = StyleSheet.create({
  welcomeLogo: {
    justifyContent: "center",
    alignSelf: "flex-end",
    width: 130,
    height: 150,
    resizeMode: "contain",
    flexDirection: "row",
  },
  MenuButton: {
    flexDirection: "row",
    alignSelf: "flex-start",
    color: AppColors.secondaryColor,
    marginTop: -175,
    justifyContent: "flex-start",
  },
  container: {
    marginTop: 190,
  },
  title: {
    color: AppColors.white,
    justifyContent: "center",
    alignSelf: "center",
    width: "30",
  },
});

export default MainScreen;
