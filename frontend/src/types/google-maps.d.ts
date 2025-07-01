declare global {
  interface Window {
    google: typeof google;
    initMap?: () => void;
  }
}

declare namespace google {
  namespace maps {
    class Map {
      constructor(mapDiv: Element | null, opts?: MapOptions);
    }
    
    interface MapOptions {
      center?: LatLng | LatLngLiteral;
      zoom?: number;
      mapTypeId?: MapTypeId;
    }
    
    class LatLng {
      constructor(lat: number, lng: number);
      lat(): number;
      lng(): number;
    }
    
    interface LatLngLiteral {
      lat: number;
      lng: number;
    }
    
    enum MapTypeId {
      HYBRID = 'hybrid',
      ROADMAP = 'roadmap',
      SATELLITE = 'satellite',
      TERRAIN = 'terrain'
    }

    class Geocoder {
      constructor();
      geocode(
        request: GeocoderRequest,
        callback: (
          results: GeocoderResult[],
          status: GeocoderStatus
        ) => void
      ): void;
    }

    interface GeocoderRequest {
      address?: string;
      location?: LatLng | LatLngLiteral;
      placeId?: string;
      bounds?: LatLngBounds;
      componentRestrictions?: GeocoderComponentRestrictions;
      region?: string;
    }

    interface GeocoderResult {
      address_components: GeocoderAddressComponent[];
      formatted_address: string;
      geometry: GeocoderGeometry;
      place_id: string;
      types: string[];
    }

    interface GeocoderAddressComponent {
      long_name: string;
      short_name: string;
      types: string[];
    }

    interface GeocoderGeometry {
      location: LatLng;
      location_type: GeocoderLocationType;
      viewport: LatLngBounds;
      bounds?: LatLngBounds;
    }

    enum GeocoderStatus {
      OK = 'OK',
      ZERO_RESULTS = 'ZERO_RESULTS',
      OVER_QUERY_LIMIT = 'OVER_QUERY_LIMIT',
      REQUEST_DENIED = 'REQUEST_DENIED',
      INVALID_REQUEST = 'INVALID_REQUEST',
      UNKNOWN_ERROR = 'UNKNOWN_ERROR'
    }

    enum GeocoderLocationType {
      ROOFTOP = 'ROOFTOP',
      RANGE_INTERPOLATED = 'RANGE_INTERPOLATED',
      GEOMETRIC_CENTER = 'GEOMETRIC_CENTER',
      APPROXIMATE = 'APPROXIMATE'
    }

    interface GeocoderComponentRestrictions {
      administrativeArea?: string;
      country?: string;
      locality?: string;
      postalCode?: string;
      route?: string;
    }

    class LatLngBounds {
      constructor(sw?: LatLng, ne?: LatLng);
      contains(latLng: LatLng): boolean;
      equals(other: LatLngBounds): boolean;
      extend(point: LatLng): LatLngBounds;
      getCenter(): LatLng;
      getNorthEast(): LatLng;
      getSouthWest(): LatLng;
      intersects(other: LatLngBounds): boolean;
      isEmpty(): boolean;
      toJSON(): LatLngBoundsLiteral;
      toString(): string;
      toUrlValue(precision?: number): string;
      union(other: LatLngBounds): LatLngBounds;
    }

    interface LatLngBoundsLiteral {
      east: number;
      north: number;
      south: number;
      west: number;
    }
  }
  
  namespace maps.places {
    class PlacesService {
      constructor(attrContainer: HTMLDivElement | Map);
      nearbySearch(
        request: PlaceSearchRequest,
        callback: (
          results: PlaceResult[] | null,
          status: PlacesServiceStatus
        ) => void
      ): void;
      getDetails(
        request: PlaceDetailsRequest,
        callback: (
          result: PlaceResult | null,
          status: PlacesServiceStatus
        ) => void
      ): void;
    }
    
    interface PlaceSearchRequest {
      location: LatLng | LatLngLiteral;
      radius: number;
      type?: string;
      keyword?: string;
      minPriceLevel?: number;
      maxPriceLevel?: number;
      openNow?: boolean;
    }

    interface PlaceDetailsRequest {
      placeId: string;
      fields?: string[];
      sessionToken?: AutocompleteSessionToken;
    }
    
    interface PlaceResult {
      place_id?: string;
      name?: string;
      vicinity?: string;
      formatted_address?: string;
      rating?: number;
      user_ratings_total?: number;
      geometry?: PlaceGeometry;
      photos?: PlacePhoto[];
      types?: string[];
      formatted_phone_number?: string;
      international_phone_number?: string;
      website?: string;
      opening_hours?: PlaceOpeningHours;
      price_level?: number;
      reviews?: PlaceReview[];
      url?: string;
      utc_offset_minutes?: number;
    }

    interface PlaceGeometry {
      location: LatLng;
      viewport?: LatLngBounds;
    }

    interface PlaceOpeningHours {
      open_now?: boolean;
      periods?: PlaceOpeningHoursPeriod[];
      weekday_text?: string[];
    }

    interface PlaceOpeningHoursPeriod {
      close?: PlaceOpeningHoursTime;
      open: PlaceOpeningHoursTime;
    }

    interface PlaceOpeningHoursTime {
      day: number;
      time: string;
      hours?: number;
      minutes?: number;
      nextDate?: number;
    }

    interface PlaceReview {
      author_name: string;
      author_url?: string;
      language: string;
      profile_photo_url: string;
      rating: number;
      relative_time_description: string;
      text: string;
      time: number;
    }
    
    interface PlacePhoto {
      height: number;
      width: number;
      photo_reference: string;
      getUrl(opts?: PhotoOptions): string;
    }

    interface PhotoOptions {
      maxHeight?: number;
      maxWidth?: number;
    }

    class AutocompleteSessionToken {
      constructor();
    }
    
    enum PlacesServiceStatus {
      OK = 'OK',
      ZERO_RESULTS = 'ZERO_RESULTS',
      OVER_QUERY_LIMIT = 'OVER_QUERY_LIMIT',
      REQUEST_DENIED = 'REQUEST_DENIED',
      INVALID_REQUEST = 'INVALID_REQUEST',
      NOT_FOUND = 'NOT_FOUND',
      ERROR = 'ERROR'
    }
  }
}

export {};