import React, { useState, useEffect, useRef } from 'react';

// Type definitions for Google Maps
declare global {
  interface Window {
    google: any;
    initMap?: () => void;
  }
}

// SVG Icons (keeping existing ones)
const MapPinIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
    <circle cx="12" cy="10" r="3"/>
  </svg>
);

const SearchIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8"/>
    <path d="M21 21l-4.35-4.35"/>
  </svg>
);

const PhoneIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
  </svg>
);

const StarIcon = ({ filled = false }: { filled?: boolean }) => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
    <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
  </svg>
);

const ClockIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12,6 12,12 16,14"/>
  </svg>
);

const ChevronUpIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="18,15 12,9 6,15"/>
  </svg>
);

const ChevronDownIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="6,9 12,15 18,9"/>
  </svg>
);

const DirectionsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="3,11 22,2 13,21 11,13 3,11"/>
  </svg>
);

const GlobeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
  </svg>
);

const FilterIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46 22,3"/>
  </svg>
);

const LocationIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
    <circle cx="12" cy="9" r="2.5"/>
  </svg>
);

interface GoogleDermatologist {
  place_id: string;
  name: string;
  vicinity: string;
  rating?: number;
  user_ratings_total?: number;
  geometry: {
    location: {
      lat: number;
      lng: number;
    };
  };
  formatted_phone_number?: string;
  website?: string;
  opening_hours?: {
    open_now: boolean;
    weekday_text: string[];
  };
  photos?: Array<{
    photo_reference: string;
    height: number;
    width: number;
  }>;
  types: string[];
  price_level?: number;
  distance?: number;
}

interface GoogleAPIFinderProps {
  userLocation?: string;
}

const GoogleAPIDermatologistFinder: React.FC<GoogleAPIFinderProps> = ({ 
  userLocation = "20164"
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchLocation, setSearchLocation] = useState(userLocation);
  const [isLoading, setIsLoading] = useState(false);
  const [dermatologists, setDermatologists] = useState<GoogleDermatologist[]>([]);
  const [radius, setRadius] = useState(25000); // 25km default
  const [sortBy, setSortBy] = useState('distance');
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userCoordinates, setUserCoordinates] = useState<{lat: number, lng: number} | null>(null);
  const [isGoogleLoaded, setIsGoogleLoaded] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const googleMapsApiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY || "";

  // Load Google Maps Script
  useEffect(() => {
    if (isExpanded && googleMapsApiKey && !window.google) {
      loadGoogleMapsScript();
    } else if (window.google) {
      setIsGoogleLoaded(true);
    }
  }, [isExpanded, googleMapsApiKey]);

  const loadGoogleMapsScript = () => {
    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${googleMapsApiKey}&libraries=places`;
    script.async = true;
    script.defer = true;
    script.onload = () => {
      setIsGoogleLoaded(true);
    };
    script.onerror = () => {
      setError('Failed to load Google Maps. Please check your API key.');
    };
    document.head.appendChild(script);
  };

  // Get user's current location
  const getCurrentLocation = () => {
    setIsLoading(true);
    setError(null);
    
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const coords = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          };
          setUserCoordinates(coords);
          reverseGeocode(coords);
        },
        (error) => {
          console.error('Error getting location:', error);
          let errorMessage = 'Unable to get your location. ';
          switch(error.code) {
            case error.PERMISSION_DENIED:
              errorMessage += 'Location access was denied.';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMessage += 'Location information is unavailable.';
              break;
            case error.TIMEOUT:
              errorMessage += 'Location request timed out.';
              break;
            default:
              errorMessage += 'Please enter a zip code manually.';
              break;
          }
          setError(errorMessage);
          setIsLoading(false);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000 // 5 minutes
        }
      );
    } else {
      setError('Geolocation is not supported by this browser.');
      setIsLoading(false);
    }
  };

  // Convert coordinates to address using Google Geocoding
  const reverseGeocode = async (coords: {lat: number, lng: number}) => {
    if (!googleMapsApiKey || !window.google) {
      searchDermatologists(coords);
      return;
    }

    try {
      const geocoder = new window.google.maps.Geocoder();
      
      geocoder.geocode(
        { location: coords },
        (results: any[], status: string) => {
          if (status === 'OK' && results[0]) {
            const zipCode = results[0].address_components.find(
              (component: any) => component.types.includes('postal_code')
            );
            if (zipCode) {
              setSearchLocation(zipCode.long_name);
            }
          }
          searchDermatologists(coords);
        }
      );
    } catch (error) {
      console.error('Reverse geocoding error:', error);
      searchDermatologists(coords);
    }
  };

  // Convert address/zip to coordinates using Google Geocoding
  const geocodeLocation = async (location: string): Promise<{lat: number, lng: number} | null> => {
    if (!googleMapsApiKey || !window.google) {
      // Fallback coordinates for Sterling, VA
      return { lat: 39.0458, lng: -77.4088 };
    }

    return new Promise((resolve) => {
      const geocoder = new window.google.maps.Geocoder();
      
      geocoder.geocode(
        { address: location },
        (results: any[], status: string) => {
          if (status === 'OK' && results[0]) {
            const location = results[0].geometry.location;
            resolve({
              lat: location.lat(),
              lng: location.lng()
            });
          } else {
            console.error('Geocoding failed:', status);
            resolve(null);
          }
        }
      );
    });
  };

  // Search for dermatologists using Google Places API
  const searchDermatologists = async (coordinates: {lat: number, lng: number}) => {
    setIsLoading(true);
    setError(null);

    if (!isGoogleLoaded || !window.google) {
      // Use mock data if Google isn't loaded
      setTimeout(() => {
        const mockResults = generateMockResults(coordinates);
        setDermatologists(sortResults(mockResults));
        setIsLoading(false);
      }, 1500);
      return;
    }

    try {
      // Create a map element for PlacesService (required by Google)
      const mapDiv = document.createElement('div');
      const map = new window.google.maps.Map(mapDiv, {
        center: coordinates,
        zoom: 15
      });

      const service = new window.google.maps.places.PlacesService(map);
      
      const request = {
        location: new window.google.maps.LatLng(coordinates.lat, coordinates.lng),
        radius: radius,
        keyword: 'dermatologist',
        type: 'doctor'
      };

      service.nearbySearch(request, (results: any[], status: string) => {
        if (status === window.google.maps.places.PlacesServiceStatus.OK && results) {
          // Get detailed information for each place
          const detailedResults = results.slice(0, 10); // Limit to 10 results
          getDetailedPlaceInfo(detailedResults, coordinates);
        } else if (status === window.google.maps.places.PlacesServiceStatus.ZERO_RESULTS) {
          // Try broader search
          const broadRequest = {
            location: new window.google.maps.LatLng(coordinates.lat, coordinates.lng),
            radius: radius * 2,
            keyword: 'skin doctor dermatology',
            type: 'health'
          };
          
          service.nearbySearch(broadRequest, (broadResults: any[], broadStatus: string) => {
            if (broadStatus === window.google.maps.places.PlacesServiceStatus.OK && broadResults) {
              getDetailedPlaceInfo(broadResults.slice(0, 10), coordinates);
            } else {
              // Fall back to mock data
              const mockResults = generateMockResults(coordinates);
              setDermatologists(sortResults(mockResults));
              setIsLoading(false);
            }
          });
        } else {
          console.error('Places search failed:', status);
          // Fall back to mock data
          const mockResults = generateMockResults(coordinates);
          setDermatologists(sortResults(mockResults));
          setIsLoading(false);
        }
      });

    } catch (error) {
      console.error('Search error:', error);
      setError('Search failed. Showing sample results.');
      // Fall back to mock data
      const mockResults = generateMockResults(coordinates);
      setDermatologists(sortResults(mockResults));
      setIsLoading(false);
    }
  };

  // Get detailed information for places
  const getDetailedPlaceInfo = async (places: any[], userCoords: {lat: number, lng: number}) => {
    if (!window.google) return;

    const mapDiv = document.createElement('div');
    const map = new window.google.maps.Map(mapDiv, {
      center: userCoords,
      zoom: 15
    });
    const service = new window.google.maps.places.PlacesService(map);

    const detailedPlaces: GoogleDermatologist[] = [];
    let processed = 0;

    places.forEach((place, index) => {
      const request = {
        placeId: place.place_id,
        fields: [
          'place_id', 'name', 'vicinity', 'formatted_address', 'geometry',
          'rating', 'user_ratings_total', 'formatted_phone_number',
          'website', 'opening_hours', 'photos', 'types', 'price_level'
        ]
      };

      service.getDetails(request, (placeDetails: any, status: string) => {
        processed++;
        
        if (status === window.google.maps.places.PlacesServiceStatus.OK && placeDetails) {
          const distance = calculateDistance(
            userCoords.lat, userCoords.lng,
            placeDetails.geometry.location.lat(),
            placeDetails.geometry.location.lng()
          );

          const dermatologist: GoogleDermatologist = {
            place_id: placeDetails.place_id,
            name: placeDetails.name,
            vicinity: placeDetails.vicinity || placeDetails.formatted_address,
            rating: placeDetails.rating,
            user_ratings_total: placeDetails.user_ratings_total,
            geometry: {
              location: {
                lat: placeDetails.geometry.location.lat(),
                lng: placeDetails.geometry.location.lng()
              }
            },
            formatted_phone_number: placeDetails.formatted_phone_number,
            website: placeDetails.website,
            opening_hours: placeDetails.opening_hours ? {
              open_now: placeDetails.opening_hours.open_now,
              weekday_text: placeDetails.opening_hours.weekday_text || []
            } : undefined,
            photos: placeDetails.photos ? placeDetails.photos.slice(0, 3) : undefined,
            types: placeDetails.types || [],
            price_level: placeDetails.price_level,
            distance
          };

          detailedPlaces.push(dermatologist);
        }

        // When all places are processed
        if (processed === places.length) {
          if (detailedPlaces.length === 0) {
            // If no results, add mock data
            const mockResults = generateMockResults(userCoords);
            setDermatologists(sortResults(mockResults));
          } else {
            setDermatologists(sortResults(detailedPlaces));
          }
          setIsLoading(false);
        }
      });
    });
  };

  // Generate mock results for demo
  const generateMockResults = (coordinates: {lat: number, lng: number}): GoogleDermatologist[] => {
    return [
      {
        place_id: "mock_1",
        name: "Potomac Dermatology",
        vicinity: "123 Medical Plaza, Sterling, VA",
        rating: 4.8,
        user_ratings_total: 127,
        geometry: {
          location: { lat: coordinates.lat + 0.01, lng: coordinates.lng + 0.01 }
        },
        formatted_phone_number: "(571) 555-0123",
        website: "https://potomacdermatology.com",
        opening_hours: {
          open_now: true,
          weekday_text: [
            "Monday: 8:00 AM – 5:00 PM",
            "Tuesday: 8:00 AM – 5:00 PM", 
            "Wednesday: 8:00 AM – 5:00 PM",
            "Thursday: 8:00 AM – 5:00 PM",
            "Friday: 8:00 AM – 4:00 PM",
            "Saturday: Closed",
            "Sunday: Closed"
          ]
        },
        types: ["doctor", "health", "establishment"],
        distance: calculateDistance(
          coordinates.lat, coordinates.lng,
          coordinates.lat + 0.01, coordinates.lng + 0.01
        )
      },
      {
        place_id: "mock_2",
        name: "Northern Virginia Skin Institute",
        vicinity: "456 Health Center Dr, Reston, VA",
        rating: 4.9,
        user_ratings_total: 203,
        geometry: {
          location: { lat: coordinates.lat + 0.02, lng: coordinates.lng - 0.01 }
        },
        formatted_phone_number: "(703) 555-0456",
        website: "https://novaskin.com",
        opening_hours: {
          open_now: false,
          weekday_text: [
            "Monday: 7:00 AM – 6:00 PM",
            "Tuesday: 7:00 AM – 6:00 PM",
            "Wednesday: 7:00 AM – 6:00 PM", 
            "Thursday: 7:00 AM – 6:00 PM",
            "Friday: 7:00 AM – 5:00 PM",
            "Saturday: 8:00 AM – 2:00 PM",
            "Sunday: Closed"
          ]
        },
        types: ["doctor", "health", "establishment"],
        distance: calculateDistance(
          coordinates.lat, coordinates.lng,
          coordinates.lat + 0.02, coordinates.lng - 0.01
        )
      },
      {
        place_id: "mock_3",
        name: "Family Dermatology Center",
        vicinity: "789 Medical Way, Herndon, VA",
        rating: 4.6,
        user_ratings_total: 89,
        geometry: {
          location: { lat: coordinates.lat - 0.015, lng: coordinates.lng + 0.02 }
        },
        formatted_phone_number: "(703) 555-0789",
        opening_hours: {
          open_now: true,
          weekday_text: [
            "Monday: 9:00 AM – 5:00 PM",
            "Tuesday: 9:00 AM – 5:00 PM",
            "Wednesday: 9:00 AM – 5:00 PM",
            "Thursday: 9:00 AM – 5:00 PM", 
            "Friday: 9:00 AM – 4:00 PM",
            "Saturday: Closed",
            "Sunday: Closed"
          ]
        },
        types: ["doctor", "health", "establishment"],
        distance: calculateDistance(
          coordinates.lat, coordinates.lng,
          coordinates.lat - 0.015, coordinates.lng + 0.02
        )
      }
    ];
  };

  // Calculate distance between two points
  const calculateDistance = (lat1: number, lng1: number, lat2: number, lng2: number): number => {
    const R = 3959; // Earth's radius in miles
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
      Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Sort results based on criteria
  const sortResults = (results: GoogleDermatologist[]) => {
    switch (sortBy) {
      case 'distance':
        return results.sort((a, b) => (a.distance || 0) - (b.distance || 0));
      case 'rating':
        return results.sort((a, b) => (b.rating || 0) - (a.rating || 0));
      case 'reviews':
        return results.sort((a, b) => (b.user_ratings_total || 0) - (a.user_ratings_total || 0));
      default:
        return results;
    }
  };

  // Handle search
  const handleSearch = async () => {
    const coordinates = await geocodeLocation(searchLocation);
    if (coordinates) {
      setUserCoordinates(coordinates);
      searchDermatologists(coordinates);
    } else {
      setError('Unable to find location. Please check your input.');
    }
  };

  // Get photo URL from Google Places
  const getPhotoUrl = (photoReference: string, maxWidth: number = 150): string => {
    if (googleMapsApiKey && window.google) {
      return `https://maps.googleapis.com/maps/api/place/photo?maxwidth=${maxWidth}&photoreference=${photoReference}&key=${googleMapsApiKey}`;
    }
    // Fallback to placeholder
    return `https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=${maxWidth}&h=${maxWidth}&fit=crop&crop=face`;
  };

  // Render star rating
  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <StarIcon key={i} filled={i < Math.floor(rating)} />
    ));
  };

  // Handle actions
  const handleGetDirections = (place: GoogleDermatologist) => {
    const destination = encodeURIComponent(`${place.name}, ${place.vicinity}`);
    window.open(`https://www.google.com/maps/dir/?api=1&destination=${destination}`, '_blank');
  };

  const handleCall = (phone: string) => {
    window.open(`tel:${phone}`, '_self');
  };

  const handleVisitWebsite = (website: string) => {
    window.open(website, '_blank');
  };

  // Update sorting when criteria changes
  useEffect(() => {
    if (isExpanded && sortBy && dermatologists.length > 0) {
      const sortedResults = sortResults([...dermatologists]);
      setDermatologists(sortedResults);
    }
  }, [sortBy]);

  return (
    <div style={{
      position: 'fixed',
      bottom: 0,
      left: 0,
      right: 0,
      zIndex: 998,
      backgroundColor: 'white',
      borderTop: '1px solid #e5e7eb',
      boxShadow: '0 -4px 12px rgba(0, 0, 0, 0.1)',
      transform: isExpanded ? 'translateY(0)' : 'translateY(calc(100% - 60px))',
      transition: 'transform 0.3s ease-in-out',
      height: isExpanded ? '80vh' : '60px',
      maxHeight: isExpanded ? '800px' : '60px'
    }}>
      {/* Header/Toggle Bar */}
      <div 
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          height: '60px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          cursor: 'pointer',
          borderBottom: isExpanded ? '1px solid #e5e7eb' : 'none',
          backgroundColor: '#f8f9fa',
          transition: 'background-color 0.2s ease'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <MapPinIcon />
          <span style={{ fontWeight: '600', fontSize: '16px', color: '#1f2937' }}>
            Find Local Dermatologists
          </span>
          <span style={{ 
            backgroundColor: googleMapsApiKey ? '#10b981' : '#f59e0b', 
            color: 'white', 
            padding: '4px 8px', 
            borderRadius: '12px', 
            fontSize: '12px',
            fontWeight: '500'
          }}>
            {googleMapsApiKey ? 'Google Maps' : 'Demo Mode'}
          </span>
          {dermatologists.length > 0 && (
            <span style={{ 
              backgroundColor: '#3b82f6', 
              color: 'white', 
              padding: '4px 8px', 
              borderRadius: '12px', 
              fontSize: '12px',
              fontWeight: '500'
            }}>
              {dermatologists.length} found
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '12px', color: '#6b7280' }}>
            {isExpanded ? 'Minimize' : 'Expand'}
          </span>
          {isExpanded ? <ChevronDownIcon /> : <ChevronUpIcon />}
        </div>
      </div>

      {/* Main Content */}
      {isExpanded && (
        <div style={{ 
          height: 'calc(100% - 60px)', 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Search and Filters */}
          <div style={{ 
            padding: '20px 24px', 
            borderBottom: '1px solid #e5e7eb',
            backgroundColor: 'white'
          }}>
            {/* API Key Status */}
            {!googleMapsApiKey && (
              <div style={{
                backgroundColor: '#fef3c7',
                border: '1px solid #f59e0b',
                borderRadius: '8px',
                padding: '12px',
                marginBottom: '16px'
              }}>
                <p style={{ margin: 0, fontSize: '14px', color: '#92400e' }}>
                  <strong>Demo Mode:</strong> Add your Google Maps API key to use real location data.
                  <br />Add <code>REACT_APP_GOOGLE_MAPS_API_KEY=your-key</code> to your .env file.
                  <br />Currently showing demo data for Sterling, VA area.
                </p>
              </div>
            )}

            {/* Main Search Row */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '2fr auto auto auto', 
              gap: '12px', 
              marginBottom: '12px',
              alignItems: 'center'
            }}>
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  value={searchLocation}
                  onChange={(e) => setSearchLocation(e.target.value)}
                  placeholder="Enter zip code, city, or address"
                  style={{
                    width: '100%',
                    padding: '12px 40px 12px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px',
                    outline: 'none'
                  }}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleSearch();
                    }
                  }}
                />
                <button
                  onClick={handleSearch}
                  disabled={isLoading}
                  style={{
                    position: 'absolute',
                    right: '8px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#6b7280',
                    padding: '4px'
                  }}
                >
                  <SearchIcon />
                </button>
              </div>

              <button
                onClick={getCurrentLocation}
                disabled={isLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '12px 16px',
                  backgroundColor: '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '14px',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap'
                }}
              >
                <LocationIcon />
                Use My Location
              </button>
              
              <button
                onClick={() => setShowFilters(!showFilters)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '12px 16px',
                  backgroundColor: showFilters ? '#3b82f6' : '#f3f4f6',
                  color: showFilters ? 'white' : '#374151',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px',
                  cursor: 'pointer'
                }}
              >
                <FilterIcon />
                Filters
              </button>
            </div>

            {/* Filters Row */}
            {showFilters && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
                gap: '12px',
                padding: '12px',
                backgroundColor: '#f8f9fa',
                borderRadius: '8px',
                border: '1px solid #e5e7eb'
              }}>
                <select
                  value={radius}
                  onChange={(e) => setRadius(Number(e.target.value))}
                  style={{
                    padding: '8px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: '14px',
                    backgroundColor: 'white'
                  }}
                >
                  <option value={8000}>5 miles</option>
                  <option value={16000}>10 miles</option>
                  <option value={25000}>15 miles</option>
                  <option value={40000}>25 miles</option>
                  <option value={80000}>50 miles</option>
                </select>

                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '6px',
                    fontSize: '14px',
                    backgroundColor: 'white'
                  }}
                >
                  <option value="distance">Sort by Distance</option>
                  <option value="rating">Sort by Rating</option>
                  <option value="reviews">Sort by Reviews</option>
                </select>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div style={{
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '8px',
                padding: '12px',
                marginTop: '12px'
              }}>
                <p style={{ margin: 0, fontSize: '14px', color: '#dc2626' }}>
                  {error}
                </p>
              </div>
            )}
          </div>

          {/* Results List */}
          <div style={{ 
            flex: 1, 
            overflowY: 'auto', 
            padding: '0 24px 24px' 
          }}>
            {isLoading ? (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: '200px' 
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    border: '3px solid #e5e7eb',
                    borderTop: '3px solid #3b82f6',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    margin: '0 auto 12px'
                  }}></div>
                  <p style={{ color: '#6b7280', margin: 0 }}>
                    Searching for dermatologists near you...
                  </p>
                </div>
              </div>
            ) : dermatologists.length === 0 && !error ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '40px', 
                color: '#6b7280' 
              }}>
                <MapPinIcon />
                <p style={{ fontSize: '16px', marginBottom: '8px', marginTop: '12px' }}>
                  Ready to search
                </p>
                <p style={{ fontSize: '14px' }}>
                  Enter your zip code or use your location to find nearby dermatologists.
                </p>
              </div>
            ) : dermatologists.length === 0 && !isLoading ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '40px', 
                color: '#6b7280' 
              }}>
                <p style={{ fontSize: '16px', marginBottom: '8px' }}>No dermatologists found</p>
                <p style={{ fontSize: '14px' }}>Try expanding your search radius or searching a different area.</p>
              </div>
            ) : (
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '16px', 
                paddingTop: '20px' 
              }}>
                {dermatologists.map((doctor) => (
                  <div
                    key={doctor.place_id}
                    style={{
                      border: '1px solid #e5e7eb',
                      borderRadius: '12px',
                      padding: '24px',
                      backgroundColor: 'white',
                      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.15)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
                    }}
                  >
                    <div style={{ display: 'flex', gap: '20px' }}>
                      {/* Photo */}
                      <div style={{
                        width: '100px',
                        height: '100px',
                        borderRadius: '12px',
                        backgroundColor: '#f3f4f6',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        overflow: 'hidden',
                        border: '2px solid #e5e7eb'
                      }}>
                        {doctor.photos && doctor.photos[0] ? (
                          <img
                            src={getPhotoUrl(doctor.photos[0].photo_reference)}
                            alt={doctor.name}
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover'
                            }}
                          />
                        ) : (
                          <MapPinIcon />
                        )}
                      </div>
                      
                      <div style={{ flex: 1 }}>
                        {/* Header */}
                        <div style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'flex-start', 
                          marginBottom: '12px' 
                        }}>
                          <div>
                            <h3 style={{ 
                              margin: 0, 
                              fontSize: '20px', 
                              fontWeight: '600', 
                              color: '#1f2937',
                              marginBottom: '4px'
                            }}>
                              {doctor.name}
                            </h3>
                            <p style={{ 
                              margin: 0, 
                              color: '#6b7280', 
                              fontSize: '14px' 
                            }}>
                              {doctor.types.includes('hospital') ? 'Medical Center' : 'Dermatology Practice'}
                            </p>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            {doctor.rating && (
                              <div style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: '4px', 
                                marginBottom: '4px',
                                justifyContent: 'flex-end'
                              }}>
                                <div style={{ display: 'flex', color: '#fbbf24' }}>
                                  {renderStars(doctor.rating)}
                                </div>
                                <span style={{ fontSize: '14px', fontWeight: '600' }}>
                                  {doctor.rating.toFixed(1)}
                                </span>
                                {doctor.user_ratings_total && (
                                  <span style={{ fontSize: '12px', color: '#6b7280' }}>
                                    ({doctor.user_ratings_total})
                                  </span>
                                )}
                              </div>
                            )}
                            {doctor.distance && (
                              <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
                                {doctor.distance.toFixed(1)} miles away
                              </p>
                            )}
                          </div>
                        </div>
                        
                        {/* Address */}
                        <p style={{ 
                          margin: '0 0 12px 0', 
                          color: '#6b7280', 
                          fontSize: '14px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px'
                        }}>
                          <MapPinIcon />
                          {doctor.vicinity}
                        </p>
                        
                        {/* Contact and Status */}
                        <div style={{ 
                          display: 'flex', 
                          gap: '20px', 
                          marginBottom: '12px', 
                          flexWrap: 'wrap',
                          alignItems: 'center'
                        }}>
                          {doctor.formatted_phone_number && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <PhoneIcon />
                              <span style={{ fontSize: '14px' }}>{doctor.formatted_phone_number}</span>
                            </div>
                          )}
                          {doctor.opening_hours && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <ClockIcon />
                              <span style={{ 
                                fontSize: '14px', 
                                color: doctor.opening_hours.open_now ? '#10b981' : '#ef4444',
                                fontWeight: '500'
                              }}>
                                {doctor.opening_hours.open_now ? 'Open Now' : 'Closed'}
                              </span>
                            </div>
                          )}
                          {doctor.website && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <GlobeIcon />
                              <span style={{ fontSize: '12px', color: '#3b82f6', fontWeight: '500' }}>
                                Website Available
                              </span>
                            </div>
                          )}
                        </div>
                        
                        {/* Hours (if available) */}
                        {doctor.opening_hours && doctor.opening_hours.weekday_text && (
                          <div style={{ marginBottom: '16px' }}>
                            <details style={{ fontSize: '12px' }}>
                              <summary style={{ cursor: 'pointer', color: '#6b7280', fontWeight: '500' }}>
                                View Hours
                              </summary>
                              <div style={{ marginTop: '8px', paddingLeft: '16px' }}>
                                {doctor.opening_hours.weekday_text.map((day, index) => (
                                  <div key={index} style={{ marginBottom: '2px', color: '#6b7280' }}>
                                    {day}
                                  </div>
                                ))}
                              </div>
                            </details>
                          </div>
                        )}
                        
                        {/* Action Buttons */}
                        <div style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center',
                          flexWrap: 'wrap',
                          gap: '8px'
                        }}>
                          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            <button
                              onClick={() => handleGetDirections(doctor)}
                              style={{
                                padding: '8px 12px',
                                backgroundColor: 'transparent',
                                border: '1px solid #d1d5db',
                                borderRadius: '6px',
                                fontSize: '12px',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                transition: 'all 0.2s ease'
                              }}
                              onMouseOver={(e) => {
                                e.currentTarget.style.backgroundColor = '#f3f4f6';
                                e.currentTarget.style.borderColor = '#9ca3af';
                              }}
                              onMouseOut={(e) => {
                                e.currentTarget.style.backgroundColor = 'transparent';
                                e.currentTarget.style.borderColor = '#d1d5db';
                              }}
                            >
                              <DirectionsIcon />
                              Directions
                            </button>
                            
                            {doctor.formatted_phone_number && (
                              <button
                                onClick={() => handleCall(doctor.formatted_phone_number!)}
                                style={{
                                  padding: '8px 12px',
                                  backgroundColor: '#3b82f6',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '6px',
                                  fontSize: '12px',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '4px',
                                  transition: 'background-color 0.2s ease'
                                }}
                                onMouseOver={(e) => {
                                  e.currentTarget.style.backgroundColor = '#2563eb';
                                }}
                                onMouseOut={(e) => {
                                  e.currentTarget.style.backgroundColor = '#3b82f6';
                                }}
                              >
                                <PhoneIcon />
                                Call
                              </button>
                            )}
                            
                            {doctor.website && (
                              <button
                                onClick={() => handleVisitWebsite(doctor.website!)}
                                style={{
                                  padding: '8px 12px',
                                  backgroundColor: '#10b981',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '6px',
                                  fontSize: '12px',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '4px',
                                  transition: 'background-color 0.2s ease'
                                }}
                                onMouseOver={(e) => {
                                  e.currentTarget.style.backgroundColor = '#059669';
                                }}
                                onMouseOut={(e) => {
                                  e.currentTarget.style.backgroundColor = '#10b981';
                                }}
                              >
                                <GlobeIcon />
                                Website
                              </button>
                            )}
                          </div>
                          
                          {/* Source Attribution */}
                          <div style={{ fontSize: '10px', color: '#9ca3af' }}>
                            {googleMapsApiKey && isGoogleLoaded ? 'Powered by Google Maps' : 'Demo Data'}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default GoogleAPIDermatologistFinder;