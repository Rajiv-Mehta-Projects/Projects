import React from 'react';
import { Text,StyleSheet } from 'react-native';

import AppColors from '../config/AppColors';

function AppError({children}) {
    return (
        <Text style = {styles.text}> {children} </Text>
    );
}

const styles = StyleSheet.create({
    text:{
        fontSize:15,
        padding:5,
        fontFamily: 'sans-serif-medium',
        color: AppColors.errorColor,
        alignSelf: 'center'
    },
})

export default AppError;