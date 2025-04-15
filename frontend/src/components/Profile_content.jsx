import React,{useState,useEffect} from 'react'

function Profile_content() {
    const [employee, setEmployee] = useState(null);

    useEffect(() => {
        const fetchEmployee = async () => {
          try {
            const res = await axios.get('http://localhost:5000/api/employee/profile');
            setEmployee(res.data);
          } catch (err) {
            console.error('Error fetching employee profile:', err);
          }
        };
    
        fetchEmployee();
      }, []);
    
      if (!employee) {
        return <div className="p-4">Loading profile...</div>;
      }
  return (
        <div className="absolute top-0 left-0 p-15">
            <div className="flex flex-col items-start">
            <img
            src={employee.image || "URL"}
            alt="Employee"
            className="w-48 h-48 object-cover rounded-xl shadow-md"
            />
            <div className="mt-4">
            <h2 className="text-xl font-semibold flex">Name of the Employee</h2>
            <p className="text-gray-600 flex">Employee ID</p>
            </div>
        </div>
    </div>
    
  )
}

export default Profile_content