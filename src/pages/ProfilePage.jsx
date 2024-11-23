import React from "react";
import { useParams, useLocation } from "react-router-dom";
import { User, Mail, Phone, MapPin, Building } from "lucide-react";

const ProfilePage = () => {
  const { id } = useParams();
  const location = useLocation();
  const userData = {
    name: "Ajay C",
    email: "23b107@psgitech.ac.in",
    phone: "9943457438",
    location: "Coimbatore,Tamil Nadu",
    department: "Engineering",
    image: location.state?.image || "/public/dataset/2.jpg",
  };

  return (
    <div className="min-h-screen bg-white text-gray-900 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl overflow-hidden shadow-xl border border-gray-200">
          {/* Profile Header */}
          <div className="relative h-48 bg-gradient-to-r from-blue-500 to-blue-600">
            <div className="absolute -bottom-20 left-1/2 transform -translate-x-1/2">
              <img
                src={userData.image}
                alt={userData.name}
                className="w-40 h-40 rounded-full border-4 border-white object-cover shadow-lg"
              />
            </div>
          </div>

          {/* Profile Content */}
          <div className="pt-24 pb-8 px-8">
            <h1 className="text-3xl font-bold text-center mb-2">
              {userData.name}
            </h1>
            <p className="text-gray-500 text-center mb-8">Employee ID: {id}</p>

            <div className="space-y-4 max-w-lg mx-auto">
              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <User className="w-6 h-6 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-500">Full Name</p>
                  <p className="font-medium">{userData.name}</p>
                </div>
              </div>

              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <Mail className="w-6 h-6 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-500">Email</p>
                  <p className="font-medium">{userData.email}</p>
                </div>
              </div>

              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <Building className="w-6 h-6 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-500">Department</p>
                  <p className="font-medium">{userData.department}</p>
                </div>
              </div>

              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <Phone className="w-6 h-6 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-500">Phone</p>
                  <p className="font-medium">{userData.phone}</p>
                </div>
              </div>

              <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <MapPin className="w-6 h-6 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-500">Location</p>
                  <p className="font-medium">{userData.location}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
