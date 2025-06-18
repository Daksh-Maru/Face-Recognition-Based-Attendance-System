import React, { useState,useEffect } from 'react'
import Employee_Table from './Employee_Table';
import Profile_content from './Profile_content';

import { useParams } from 'react-router-dom';

function Profile() {
  const { employeeId } = useParams();
  
  return (
    <div>
      <Profile_content employeeId={employeeId}/>
      <Employee_Table employeeId={employeeId} />
    </div>
  );
}
export default Profile
